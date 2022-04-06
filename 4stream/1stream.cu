#include "cuda_runtime.h"  
#include <iostream>
#include <stdio.h>  
#include <math.h>  
 
#define N (1024*1024)  //每次从CPU传输到GPU的数据块大小
#define FULL_DATA_SIZE N*20  //总数据量
 
__global__ void kernel(int* a, int *b, int*c)
{
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	//这里每次计算N个数组
	if (threadID < N)
	{
		c[threadID] = (a[threadID] + b[threadID]) / 2;
	}
}
//使用单流：目的：计算两个数组，数组大小均为FULL_DATA_SIZE，的和
int main()
{
	//获取设备属性
	cudaDeviceProp prop;
	int deviceID;
	cudaGetDevice(&deviceID);
	cudaGetDeviceProperties(&prop, deviceID);
 
	//检查设备是否支持重叠功能，不支持则不能使用多流加速
	if (!prop.deviceOverlap)
	{
		printf("No device will handle overlaps. so no speed up from stream.\n");
		return 0;
	}
 
	//启动计时器
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
 
	//创建一个CUDA stream
	cudaStream_t stream;
	cudaStreamCreate(&stream);
 
	int *host_a, *host_b, *host_c;
	int *dev_a, *dev_b, *dev_c;
 
	//在GPU上分配内存： GPU上分配的内存大小是N
	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));
 
	//在CPU上分配：页锁定内存，使用流的时候，要使用页锁定内存
	cudaMallocHost((void**)&host_a, FULL_DATA_SIZE * sizeof(int));
	cudaMallocHost((void**)&host_b, FULL_DATA_SIZE * sizeof(int));
	cudaMallocHost((void**)&host_c, FULL_DATA_SIZE * sizeof(int));
 
	//主机上的内存赋值
	for (int i = 0; i < FULL_DATA_SIZE; i++) {
		host_a[i] = i;
		host_b[i] = FULL_DATA_SIZE - i;
	}
	// 内存数据能够在分块传输的同时，GPU也在执行核函数运算，这样的异步操作，可以提升性能
	// 将输入缓冲区划分为更小的块，每次向GPU copy N块数据，在stream上执行。并在每个块上执行“数据传输到GPU”，“计算”，“数据传输回CPU”三个步骤
	for (int i = 0; i < FULL_DATA_SIZE; i += N) {
		//异步将host数据copy到device并执行kernel函数
		//因为这个操作是异步的，所以在copy数据的时候，kernel函数就可以开始执行。也就是边copy边执行计算
		//比如第一个N块数据拷贝完了，kernel函数就计算，这时第二个N块数据同时进行拷贝。但是如果是没有stream，
		//就必须要等到所有数据全部拷贝完再执行计算，这么做可以提高性能
		cudaMemcpyAsync(dev_a, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(dev_b, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);
		//注意这里开启线程数是N
		kernel <<<N / 1024, 1024, 0, stream >>> (dev_a, dev_b, dev_c);
 
		cudaMemcpyAsync(host_c + i, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost, stream);
	}
 
	// wait until gpu execution finish  
	cudaStreamSynchronize(stream);
 
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
 
	std::cout << "消耗时间GPU： " << elapsedTime <<"ms"<< std::endl;
 
	//输出前10个结果
	// for (int i = 0; i < 10; i++)
	// {
	// 	std::cout << host_c[i] << std::endl;
	// }
 
 
	// free stream and mem  
	cudaFreeHost(host_a);
	cudaFreeHost(host_b);
	cudaFreeHost(host_c);
 
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
 
	cudaStreamDestroy(stream);
	return 0;
}