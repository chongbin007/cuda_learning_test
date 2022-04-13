#include "cuda_runtime.h"
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define N (1024 * 1024)		  //每次从CPU传输到GPU的数据块大小
#define FULL_DATA_SIZE N * 20 //总数据量

__global__ void kernel(int *a, int *b, int *c)
{
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	//这里线程号应该小于FULL_DATA_SIZE
	if (threadID < FULL_DATA_SIZE)
	{
		c[threadID] = (a[threadID] + b[threadID]) / 2;
	}
}
//目的：计算两个数组，数组大小均为FULL_DATA_SIZE，的和
int main()
{

	int *host_a, *host_b, *host_c;
	int *dev_a, *dev_b, *dev_c;

	//在GPU上分配内存
	cudaMalloc((void **)&dev_a, FULL_DATA_SIZE * sizeof(int));
	cudaMalloc((void **)&dev_b, FULL_DATA_SIZE * sizeof(int));
	cudaMalloc((void **)&dev_c, FULL_DATA_SIZE * sizeof(int));

	//在CPU上分配：可分页内存
	//数组大小FULL_DATA_SIZE
	host_a = (int *)malloc(FULL_DATA_SIZE * sizeof(int));
	host_b = (int *)malloc(FULL_DATA_SIZE * sizeof(int));
	host_c = (int *)malloc(FULL_DATA_SIZE * sizeof(int));

	//主机上的两个数组随机赋值
	for (int i = 0; i < FULL_DATA_SIZE; i++)
	{
		host_a[i] = i;
		host_b[i] = FULL_DATA_SIZE - i;
	}

	// copy host to device
	cudaMemcpy(dev_a, host_a, FULL_DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, host_b, FULL_DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice);
	std::cout << "启动 " << std::endl;

	cudaDeviceSynchronize();
	//启动计时器
	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//启动函数，做数值加法
	kernel<<<FULL_DATA_SIZE / 1024, 1024>>>(dev_a, dev_b, dev_c);

	//数据拷贝回主机
	cudaMemcpy(host_c, dev_c, FULL_DATA_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	std::cout << "event计时： " << elapsedTime << "ms" << std::endl;

	cudaFreeHost(host_a);
	cudaFreeHost(host_b);
	cudaFreeHost(host_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	return 0;
}
//event计时： 111.983ms