#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define N (1024 * 1024)
#define FULL_DATA_SIZE (N * 2)

__global__ void kernel1(int *a, int *b, int *c)
{
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadID < N)
    {
        c[threadID] = (a[threadID] + b[threadID]) / 2;
    }
}
__global__ void kernel2(int *a, int *b, int *c)
{
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadID < N)
    {
        c[threadID] = (a[threadID] * b[threadID]) / 2;
    }
}

int main(void)
{
    cudaDeviceProp prop;
    int whichDevice;
    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);
    if (!prop.deviceOverlap)
    {
        printf("Device will not handle overlaps, so no speed up from streams\n");
        return 0;
    }

    //初始化两个流
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    int *host_a, *host_b, *host_c;
    int *dev_a0, *dev_b0, *dev_c0;
    int *dev_a1, *dev_b1, *dev_c1;

    //在GPU上分配内存： GPU上分配的内存大小是N
    cudaMalloc((void **)&dev_a0, N * sizeof(int));
    cudaMalloc((void **)&dev_b0, N * sizeof(int));
    cudaMalloc((void **)&dev_c0, N * sizeof(int));
    cudaMalloc((void **)&dev_a1, N * sizeof(int));
    cudaMalloc((void **)&dev_b1, N * sizeof(int));
    cudaMalloc((void **)&dev_c1, N * sizeof(int));

    //在CPU上分配：页锁定内存，使用流的时候，要使用页锁定内存
    cudaHostAlloc((void **)&host_a, FULL_DATA_SIZE * sizeof(int),
                  cudaHostAllocDefault);
    cudaHostAlloc((void **)&host_b, FULL_DATA_SIZE * sizeof(int),
                  cudaHostAllocDefault);
    cudaHostAlloc((void **)&host_c, FULL_DATA_SIZE * sizeof(int),
                  cudaHostAllocDefault);

    //主机上的内存赋值
    for (int i = 0; i < FULL_DATA_SIZE; i++)
    {
        host_a[i] = rand();
        host_b[i] = rand();
    }
    cudaEvent_t start, stop;
    float elapsedTime;

    //启动计时器
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0); //在默认stream中插入start事件

    //在整体数据上循环，每个数据块的大小为N, 每次将2N个数据块传给stream
    // N个传个stream0, N个传给stream1

    //将锁定内存以异步方式复制到设备上
    cudaMemcpyAsync(dev_a0, host_a, N * sizeof(int), cudaMemcpyHostToDevice, stream0);
    cudaMemcpyAsync(dev_b0, host_b, N * sizeof(int), cudaMemcpyHostToDevice, stream0);
    kernel1<<<N / 1024, 1024, 0, stream0>>>(dev_a0, dev_b0, dev_c0);
    //将数据从设备复制回锁定内存
    cudaMemcpyAsync(host_c, dev_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream0);

    cudaMemcpyAsync(dev_a1, host_a + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(dev_b1, host_b + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
    kernel2<<<N / 1024, 1024, 0, stream1>>>(dev_a1, dev_b1, dev_c1);
    cudaMemcpyAsync(host_c + N, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream1);

    cudaEventRecord(stop, 0); //在默认中插入stop事件，默认流会同步所有stream
                              //等待event会阻塞调用host线程，同步操作，等待stop事件.
                              //该函数类似于cudaStreamSynchronize，只不过是等待一个event而不是整个stream执行完毕
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time taken: %3.1f ms\n", elapsedTime);

    //在host上检查计算的值是否正确
    //检查host_c是结果从device拷贝回来的结果，host[a]和host[b]算法是否等于host_c
    printf("info: check result in host\n");
    for (size_t i = 0; i < N; i++)
    {
        if (host_c[i] != (host_a[i] + host_b[i]) / 2)
        {
            printf("test failed.\n");
            return;
        }
    }
    for (size_t i = N; i < FULL_DATA_SIZE; i++)
    {
        if (host_c[i] != (host_a[i] * host_b[i]) / 2)
        {
            printf("test failed.\n");
            return;
        }
    }
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
    printf("test passed.\n");
    //释放流和内存
    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_c);
    cudaFree(dev_a0);
    cudaFree(dev_b0);
    cudaFree(dev_c0);
    cudaFree(dev_a1);
    cudaFree(dev_b1);
    cudaFree(dev_c1);

    return 0;
}