
#include <stdio.h>
__global__ void staticReverse(int *d, int n)
{
  __shared__ int s[1000];
  int t = threadIdx.x;
  int tr = n-t-1;
  //从global memory拷贝写入shared memory
  s[t] = d[t];
  //因为数组s是所有线程共享的，如果不做同步执行下面语句则可能出现数据竞争问题
	// Will not conttinue until all threads completed
	//调用同步函数，只有当前block中所有线程都完成之后，再往下走
  __syncthreads();
  //从shared memory读，然后写回到global memory
  d[t] = s[tr];
}

__global__ void dynamicReverse(int *d, int n)
{
  extern __shared__ int s[];
  int t = threadIdx.x;
  int tr = n-t-1;
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}
//目的：将一个数组中的数据前后交换，实现倒序
int main(void)
{
  const int n = 1000;
  int a[n], r[n], d[n];
  
  for (int i = 0; i < n; i++) {
    a[i] = i;
    r[i] = n-i-1;
    d[i] = 0;
  }

  int *d_d;
  cudaMalloc(&d_d, n * sizeof(int)); 
  
  // run version with static shared memory
  cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
  float time_gpu;
  cudaEvent_t start_GPU,stop_GPU;
  cudaEventCreate(&start_GPU);
  cudaEventCreate(&stop_GPU);
  cudaEventRecord(start_GPU,0);
  staticReverse<<<1,n>>>(d_d, n);
  //globalReverse<<<1,n>>>(d_d, n);
  cudaEventRecord(stop_GPU,0);
  cudaEventSynchronize(start_GPU);
  cudaEventSynchronize(stop_GPU);
  cudaEventElapsedTime(&time_gpu, start_GPU,stop_GPU);
  printf("\nThe time from GPU:\t%f(ms)\n", time_gpu);
  cudaDeviceSynchronize();
  cudaEventDestroy(start_GPU);
  cudaEventDestroy(stop_GPU);
  
  cudaMemcpy(d, d_d, n*sizeof(int), cudaMemcpyDeviceToHost);
  //check
  for (int i = 0; i < n; i++) {
    if (d[i] != r[i]) 
      printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);
  }
    
  
  // run dynamic shared memory version
  cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);

  cudaEventCreate(&start_GPU);
  cudaEventCreate(&stop_GPU);
  cudaEventRecord(start_GPU,0);
  dynamicReverse<<<1,n,n*sizeof(int)>>>(d_d, n);
  cudaEventRecord(stop_GPU,0);
  cudaEventSynchronize(start_GPU);
  cudaEventSynchronize(stop_GPU);
  cudaEventElapsedTime(&time_gpu, start_GPU,stop_GPU);
  printf("\nThe time from GPU:\t%f(ms)\n", time_gpu);
  cudaDeviceSynchronize();
  cudaEventDestroy(start_GPU);
  cudaEventDestroy(stop_GPU);
  cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) 
    if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);
}