#include <iostream>
#include <math.h>
// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 10000000;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  //这个Unified Memory内存空间使用cudaMallocManaged创建，该内存可以在CPU和GPU之间共享
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the GPU
  //使用1block，1个thread来执行kernel
  add<<<1, 1>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  //等GPU执行完毕，写会内存，因为上面函数是异步过程，所以如果不等GPU写完则会出现数据竞争情况。
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}
//15s