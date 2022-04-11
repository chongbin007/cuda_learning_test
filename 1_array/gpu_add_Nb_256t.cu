#include <iostream>
#include <math.h>
// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x; //线程号
  int stride = blockDim.x * gridDim.x; //总线程数
  //这里用一个for循环来处理，我们叫做grid-stride loop.
  //他是为了即使保证线程数量如果小于元素N数，可以均匀分配
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}


int main(void)
{
  int N = 10000000;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  //这个Unified Memory内存空间使用cudaMallocManaged创建，该内存可以在CPU和GPU之间共享
  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++)
  {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the GPU
  //使用1block，1个thread来执行kernel
  int blockSize = 256;                             //开启256个线程
  int numBlocks = (N + blockSize - 1) / blockSize; // block数为N/256，但是要注意四舍五入
  add<<<numBlocks, blockSize>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  //等GPU执行完毕，写会内存，因为上面函数是异步过程，所以如果不等GPU写完则会出现数据竞争情况。
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i] - 3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}
// 26.175ms