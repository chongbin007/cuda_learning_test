#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define N (1024 * 1024)
#define FULL_DATA_SIZE (N * 20)

__global__ void kernel(int *a, int *b, int *c)
{
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadID < N)
    {
        c[threadID] = (a[threadID] + b[threadID]) / 2;
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
    cudaEvent_t start, stop;
    float elapsedTime;

    //启动计时器
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0); //在stream0中插入start事件

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
        host_a[i] = i;
        host_b[i] = FULL_DATA_SIZE - i;
    }

    //在整体数据上循环，每个数据块的大小为N, 每次将2N个数据块传给stream
    // N个传个stream0, N个传给stream1
    for (int i = 0; i < FULL_DATA_SIZE; i += N * 2)
    {
        //将锁定内存以异步方式复制到设备上
        cudaMemcpyAsync(dev_a0, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(dev_b0, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0);
        kernel<<<N / 1024, 1024, 0, stream0>>>(dev_a0, dev_b0, dev_c0);
        //将数据从设备复制回锁定内存
        cudaMemcpyAsync(host_c + i, dev_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream0);

        cudaMemcpyAsync(dev_a1, host_a + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(dev_b1, host_b + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
        kernel<<<N / 1024, 1024, 0, stream1>>>(dev_a1, dev_b1, dev_c1);
        cudaMemcpyAsync(host_c + i + N, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream1);
    }

    //在停止应用程序的计时器之前，首先将两个流进行同步
    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);
    cudaEventRecord(stop, 0); //在stream0中插入stop事件
                              //等待event会阻塞调用host线程，同步操作，等待stop事件.
                              //该函数类似于cudaStreamSynchronize，只不过是等待一个event而不是整个stream执行完毕
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << "消耗时间： " << elapsedTime << "ms" << std::endl;

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
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
    return 0;
}