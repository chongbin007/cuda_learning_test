#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define N 500

__global__ void kernel1(int *a, int *b, int *c)
{
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadID < N)
    {
        c[threadID] = a[threadID] + b[threadID];
        printf("k1: %d, ", c[threadID]);
    }
}
__global__ void kernel2(int *a, int *b, int *c)
{
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadID < N)
    {
        c[threadID] = a[threadID] * b[threadID];
        printf("k2: %d, ", c[threadID]);
    }
}

__global__ void kernel_default()
{
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    printf("GPU\n");
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
    cudaStream_t streamA, streamB;
    cudaStreamCreate(&streamA);
    cudaStreamCreate(&streamB);

    int *host_a, *host_b, *host_c1, *host_c2;
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
    cudaHostAlloc((void **)&host_a, N * sizeof(int),
                  cudaHostAllocDefault);
    cudaHostAlloc((void **)&host_b, N * sizeof(int),
                  cudaHostAllocDefault);
    cudaHostAlloc((void **)&host_c1, N * sizeof(int),
                  cudaHostAllocDefault);
    cudaHostAlloc((void **)&host_c2, N * sizeof(int),
                  cudaHostAllocDefault);

    //主机上的内存赋值
    for (int i = 0; i < N; i++)
    {
        host_a[i] = i;
        host_b[i] = i;
    }
    cudaEvent_t start, stop;
    float elapsedTime;

    //启动计时器
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //在streamA 上记录event start
    cudaEventRecord(start, streamA);

    //启动两个kernel在不同的stream上
    //将锁定内存以异步方式复制到设备上
    cudaMemcpyAsync(dev_a0, host_a, N * sizeof(int), cudaMemcpyHostToDevice, streamA);
    cudaMemcpyAsync(dev_b0, host_b, N * sizeof(int), cudaMemcpyHostToDevice, streamA);
    cudaMemcpyAsync(dev_a1, host_a, N * sizeof(int), cudaMemcpyHostToDevice, streamB);
    cudaMemcpyAsync(dev_b1, host_b, N * sizeof(int), cudaMemcpyHostToDevice, streamB);
    kernel1<<<1, N, 0, streamA>>>(dev_a0, dev_b0, dev_c0);
    kernel2<<<1, N, 0, streamB>>>(dev_a1, dev_b1, dev_c1);
    cudaMemcpyAsync(host_c1, dev_c0, N * sizeof(int), cudaMemcpyDeviceToHost, streamA);
    cudaMemcpyAsync(host_c2, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, streamB);

    // 在streamB上记录event stop
    cudaEventRecord(stop, streamA); 
    //streamB等待事件stop完成，
    cudaStreamWaitEvent(streamB, stop); 
    cudaEventSynchronize(stop); //等待event会阻塞调用host线程，同步操作，等待stop事件.
                                //该函数类似于cudaStreamSynchronize，只不过是等待一个event而不是整个stream执行完毕
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time taken: %3.1f ms\n", elapsedTime);

    //在host上检查计算的值是否正确
    //检查host_c是结果从device拷贝回来的结果，host[a]和host[b]算法是否等于host_c
    //这里检查结果，结果正确，说明上面的两个stream任务都计算完成且正确
    //并且同步成功，如果报错了，说明没同步成功，有stream还没有计算完成就继续执行了。
    printf("info: check result in host\n");
    for (size_t i = 0; i < N; i++)
    {
        if (host_c1[i] != (host_a[i] + host_b[i]))
        {
            printf("test failed.\n");
            return;
        }
    }
    for (size_t i = 0; i < N; i++)
    {
        if (host_c2[i] != (host_a[i] * host_b[i]))
        {
            printf("test failed.\n");
            return;
        }
    }
    cudaStreamDestroy(streamA);
    cudaStreamDestroy(streamB);
    printf("test passed.\n");
    //释放流和内存
    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_c1);
    cudaFreeHost(host_c2);
    cudaFree(dev_a0);
    cudaFree(dev_b0);
    cudaFree(dev_c0);
    cudaFree(dev_a1);
    cudaFree(dev_b1);
    cudaFree(dev_c1);

    return 0;
}