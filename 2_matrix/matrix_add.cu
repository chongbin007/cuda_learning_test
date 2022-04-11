#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

void initialData(float *ip, int size)
{
	time_t t;
	srand((unsigned)time(&t));
	for (int i = 0; i < size; i++)
		ip[i] = i;
}

void initDevice(int devNum)
{
	int dev = devNum;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	printf("Using device %d: %s\n", dev, deviceProp.name);
	cudaSetDevice(dev);
}
void checkResult(float *hostRef, float *gpuRef, const int N)
{
	double epsilon = 1.0E-8;
	for (int i = 0; i < N; i++)
	{
		if (abs(hostRef[i] - gpuRef[i]) > epsilon)
		{
			printf("Results don\'t match!\n");
			printf("%f(hostRef[%d] )!= %f(gpuRef[%d])\n", hostRef[i], i, gpuRef[i], i);
			return;
		}
	}
	printf("Check result success!\n");
}

// CPU对照组，用于对比加速比
void sumMatrix2DonCPU(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
	float *a = MatA;
	float *b = MatB;
	float *c = MatC;
	for (int j = 0; j < ny; j++)
	{
		for (int i = 0; i < nx; i++)
		{
			c[i] = a[i] + b[i];
		}
		//一行结束后，abc指针向下移动一行
		c += nx;
		b += nx;
		a += nx;
	}
}

//核函数，每一个线程计算矩阵中的一个元素。
//因为2Dblock+2Dgrid就铺成了一张大网，我们对这个2D大网进行编号
//每一个线程在大网中的下标，就是矩阵中对应的下标
__global__ void sumMatrix(float *MatA, float *MatB, float *MatC, int nx, int ny)
{

	// ix和iy为大网唯一确定的线程索引
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;

	//得到全局唯一id
	int idx = ix + iy * nx;
	//或者
	// int idx = ix + iy *  blockDim.x * gridDim.x;
	if (ix < nx && iy < ny)
	{
		MatC[idx] = MatA[idx] + MatB[idx];
	}
}

//主函数
int main(int argc, char **argv)
{
	//设备初始化
	printf("strating...\n");
	initDevice(0);

	//输入二维矩阵
	int nx = 8192;
	int ny = 8192;
	int nBytes = nx * ny * sizeof(float);

	// Malloc，开辟主机内存
	float *A_host = (float *)malloc(nBytes);
	float *B_host = (float *)malloc(nBytes);
	float *C_host = (float *)malloc(nBytes);
	float *C_from_gpu = (float *)malloc(nBytes);
	initialData(A_host, nx * ny);
	initialData(B_host, nx * ny);

	// cudaMalloc，开辟设备内存
	float *A_dev = NULL;
	float *B_dev = NULL;
	float *C_dev = NULL;
	cudaMalloc((void **)&A_dev, nBytes);
	cudaMalloc((void **)&B_dev, nBytes);
	cudaMalloc((void **)&C_dev, nBytes);

	//输入数据从主机内存拷贝到设备内存
	cudaMemcpy(A_dev, A_host, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(B_dev, B_host, nBytes, cudaMemcpyHostToDevice);

	//二维线程块，32×32
	dim3 block(32, 32);
	//二维grid
	dim3 grid((nx - 1) / block.x + 1, (ny - 1) / block.y + 1);

	//测试GPU执行时间
	double gpuStart = cpuSecond();
	//调用GPU执行
	sumMatrix<<<grid, block>>>(A_dev, B_dev, C_dev, nx, ny);

	cudaDeviceSynchronize();
	double gpuTime = cpuSecond() - gpuStart;
	printf("GPU Execution Time: %f ms\n", gpuTime * 1000);

	//在CPU上完成相同的任务
	cudaMemcpy(C_from_gpu, C_dev, nBytes, cudaMemcpyDeviceToHost);
	double cpuStart = cpuSecond();
	sumMatrix2DonCPU(A_host, B_host, C_host, nx, ny);
	double cpuTime = cpuSecond() - cpuStart;
	printf("CPU Execution Time: %f ms\n", cpuTime * 1000);

	//检查GPU与CPU计算结果是否相同
	cudaMemcpy(C_from_gpu, C_dev, nBytes, cudaMemcpyDeviceToHost);
	checkResult(C_host, C_from_gpu, nx * ny);

	cudaFree(A_dev);
	cudaFree(B_dev);
	cudaFree(C_dev);
	free(A_host);
	free(B_host);
	free(C_host);
	free(C_from_gpu);
	return 0;
}

// GPU Execution Time: 17.311096 ms
// CPU Execution Time: 386.194944 ms
// Check result success!