#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"
#include <stdio.h>
 
using namespace std;

#define COPY_COUNTS 10
#define MEM_SIZE 25*1024*1024
 
float cuda_host_alloc_test(int size, bool up)
{
	//耗时统计
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
 
	int *a, *dev_a;
	//在主机上分配页锁定内存
    cudaError_t cudaStatus = cudaMallocHost((void **)&a, size * sizeof(*a));
	//在设备上分配内存空间
	cudaStatus = cudaMalloc((void **)&dev_a, size * sizeof(*dev_a));
	//计时开始
	cudaEventRecord(start, 0);
 
	for (int i = 0; i < COPY_COUNTS; i++)
	{
		//从主机到设备复制数据
		cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(*dev_a), cudaMemcpyHostToDevice);
		//从设备到主机复制数据
		cudaStatus = cudaMemcpy(a, dev_a, size * sizeof(*dev_a), cudaMemcpyDeviceToHost);

	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
 
	cudaFreeHost(a);
	cudaFree(dev_a);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
 
	return (float)elapsedTime / 1000;
 
}
 
float cuda_host_Malloc_test(int size, bool up)
{
	//耗时统计
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int *a, *dev_a;
 
	//在主机上分配可分页内存
	a = (int*)malloc(size * sizeof(*a));
 
	//在设备上分配内存空间
	cudaError_t	cudaStatus = cudaMalloc((void **)&dev_a, size * sizeof(*dev_a));
 
	//计时开始
	cudaEventRecord(start, 0);
 
	//执行从copy host to device 然后再 device to host执行100次，记录时间
	for (int i = 0; i < COPY_COUNTS; i++) {
		//从主机到设备复制数据
		cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(*dev_a), cudaMemcpyHostToDevice);
		//从设备到主机复制数据
		cudaStatus = cudaMemcpy(a, dev_a, size * sizeof(*dev_a), cudaMemcpyDeviceToHost);

	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
 
	free(a);
	cudaFree(dev_a);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
 
	return (float)elapsedTime / 1000;
}
 
int main()
{
	float allocTime = cuda_host_alloc_test(MEM_SIZE, true);
	cout << "页锁定内存: " << allocTime << " s" << endl;
	float mallocTime = cuda_host_Malloc_test(MEM_SIZE, true);
	cout << "可分页内存: " << mallocTime << " s" << endl;
	return 0;
}

// 反复拷贝数据进行性能测试
// 页锁定内存: 0.658992 s
// 可分页内存: 1.22233 s