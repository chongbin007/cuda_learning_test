#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define N_STREAM 20
#define N (1024*1024)
#define FULL_DATA_SIZE (N * N_STREAM)


__global__ void MyKernel(int *a, int *b, int *c){
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadID < FULL_DATA_SIZE){
       c[threadID] = (a[threadID] + b[threadID]) / 2;
    }
}

int main(void){
    cudaDeviceProp prop;
    int whichDevice;
    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);
    if (!prop.deviceOverlap){
        printf("Device will not handle overlaps, so no speed up from streams\n");
        return 0;
    }
    cudaEvent_t start, stop;
    float elapsedTime;

    //启动计时器
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0); //在stream0中插入start事件

    //初始化N_STREAM个流
    cudaStream_t stream[N_STREAM];
    for (int i = 0; i < N_STREAM; ++i)
        cudaStreamCreate(&stream[i]);

    int *host_a, *host_b, *host_c;
    int *dev_a0, *dev_b0, *dev_c0;


    //在GPU上分配内存： GPU上分配的内存大小是N
    cudaMalloc((void **)&dev_a0, FULL_DATA_SIZE * sizeof(int));
    cudaMalloc((void **)&dev_b0, FULL_DATA_SIZE * sizeof(int));
    cudaMalloc((void **)&dev_c0, FULL_DATA_SIZE * sizeof(int));


    //在CPU上分配：页锁定内存，使用流的时候，要使用页锁定内存
    cudaHostAlloc((void **)&host_a, FULL_DATA_SIZE * sizeof(int),
        cudaHostAllocDefault);
    cudaHostAlloc((void **)&host_b, FULL_DATA_SIZE * sizeof(int),
        cudaHostAllocDefault);
    cudaHostAlloc((void **)&host_c, FULL_DATA_SIZE * sizeof(int),
        cudaHostAllocDefault);

	//主机上的内存赋值
    for (int i = 0; i < FULL_DATA_SIZE; i++){
        host_a[i] = i;
        host_b[i] = FULL_DATA_SIZE - i;
    }

    //每个流计算N个数据：比如stream0计算数据0~(N-1), stream1计算数据N~(2N-1)
    for (int i = 0; i < N_STREAM; i++) {
        //host copy to device
        cudaMemcpyAsync(dev_a0 + i * N, host_a + i * N, N, cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(dev_b0 + i * N, host_b + i * N, N, cudaMemcpyHostToDevice, stream[i]);
        MyKernel <<<N / 1024, 1024, 0, stream[i]>>>(dev_a0 + i * N, dev_b0 + i * N, dev_c0 + i * N);
        cudaMemcpyAsync(host_c + i * N, dev_c0 + i * N, N, cudaMemcpyDeviceToHost, stream[i]);
    }

    //在停止应用程序的计时器之前，首先将两个流进行同步
    for (int i = 0; i < N_STREAM; ++i)
        cudaStreamSynchronize(stream[i]);
    cudaEventRecord(stop, 0);//在stream0中插入stop事件
	//等待event会阻塞调用host线程，同步操作，等待stop事件.
	//该函数类似于cudaStreamSynchronize，只不过是等待一个event而不是整个stream执行完毕
    cudaEventSynchronize(stop);

	//stop事件过来的时候，就说明
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time taken: %3.1f ms\n", elapsedTime);

    //销毁流
    for (int i = 0; i < N_STREAM; ++i)
        cudaStreamDestroy(stream[i]);
    //释放流和内存
    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_c);
    cudaFree(dev_a0);
    cudaFree(dev_b0);
    cudaFree(dev_c0);


    return 0;
}

