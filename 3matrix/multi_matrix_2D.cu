#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>


#define A_ROW_SIZE 700
#define A_COL_SIZE 300
#define B_ROW_SIZE 300
#define B_COL_SIZE 600

#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}

double cpuSecond()
{
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return((double)tp.tv_sec+(double)tp.tv_usec*1e-6);

}

void initialData(double* ip,int size)
{
  time_t t;
  srand((unsigned )time(&t));
  for(int i=0;i<size;i++)
    ip[i]=i;
}

void initDevice(int devNum)
{
  int dev = devNum;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp,dev));
  printf("Using device %d: %s\n",dev,deviceProp.name);
  CHECK(cudaSetDevice(dev));

}
void checkResult(double * hostRef,double * gpuRef,const int N)
{
  double epsilon=1.0E-8;
  for(int i=0;i<N;i++)
  {
    if(abs(hostRef[i]-gpuRef[i])>epsilon)
    {
      printf("Results don\'t match!\n");
      printf("%f(hostRef[%d] )!= %f(gpuRef[%d])\n",hostRef[i],i,gpuRef[i],i);
      return;
    }
  }
  printf("Check result success!\n");
}


//CPU对照组，用于对比加速比
//C = A*B
void MatrixMulCPU(double* _C, const double *_A,const double *_B)
{
    double sum = 0;
    for (int i = 0; i < A_COL_SIZE; ++i)
    {
        for (int j = 0; j < B_COL_SIZE; ++j)
        {
            sum = 0;
            for (int k = 0; k < A_COL_SIZE; ++k)
            {
//i*_wa得到当前对应的是A矩阵中的哪一行，+k对应当前行的哪一列．矩阵Ａ的stride是wa
//j对应当前矩阵Ｂ的哪一列，＋k*wb依次表示第０行的j列,第１行的j列...第wa-1行的j列．矩阵Ｂ的stride是wb
                sum += (double)_A[i*A_COL_SIZE+k]*(double)_B[k*B_COL_SIZE+ j];
            }
            _C[i*B_COL_SIZE+j] = (double)sum;
        }
    }
}


//核函数，每一个线程计算矩阵中的一个元素。
__global__ void sumMatrix(double * A,double * B,double * C)
{
    //因为2Dblock+2Dgrid就铺成了一张大网，我们对这个2D大网进行编号
    //ix和iy为大网唯一确定的线程索引
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;

   
    //或者
    //int idx = ix + iy *  blockDim.x * gridDim.x;
    double sum = 0;
    if(row < A_ROW_SIZE && col < B_COL_SIZE) {
        //A的某一行，乘以B的对应列，得到结果
        for (int i = 0; i < A_COL_SIZE; ++i) {
            sum += A[row*A_COL_SIZE + i] * B[i*B_COL_SIZE + col];
        }
        //结果放到C线程矩阵与C全局矩阵一一对应
        C[row*B_COL_SIZE+col] = sum;
    }
}

//主函数
int main(int argc,char** argv)
{
    //设备初始化
    printf("strating...\n");
    initDevice(0);
    //得到ABC的大小
    int N_A = A_ROW_SIZE * A_COL_SIZE;
    int N_B = B_ROW_SIZE * B_COL_SIZE;
    int N_C = A_ROW_SIZE * B_COL_SIZE;


    //主机上ABC分配内存
    double* A_host = (double*)malloc(N_A * sizeof(double));
    double* B_host = (double*)malloc(N_B * sizeof(double));
    double* C_host = (double*)malloc(N_C * sizeof(double));

    initialData(A_host, N_A);
    initialData(B_host, N_B);

    //开辟设备内存
    double* A_dev = NULL;
    double* B_dev = NULL;
    double* C_dev = NULL;
    CHECK(cudaMalloc((void**)&A_dev, N_A * sizeof(double)));
    CHECK(cudaMalloc((void**)&B_dev, N_B * sizeof(double)));
    CHECK(cudaMalloc((void**)&C_dev, N_C * sizeof(double)));

    //输入数据从主机内存拷贝到设备内存
    CHECK(cudaMemcpy(A_dev, A_host, N_A * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_dev, B_host, N_B * sizeof(double), cudaMemcpyHostToDevice));

    //二维线程块，32×32
    dim3 block(32, 32);
    //二维grid
    dim3 grid((A_ROW_SIZE-1)/block.x+1, (B_COL_SIZE-1)/block.y+1);


    //测试GPU执行时间
    double gpuStart = cpuSecond();
    //调用GPU执行
    sumMatrix<<<grid,block>>>(A_dev, B_dev, C_dev);

    CHECK(cudaDeviceSynchronize());
    double gpuTime = cpuSecond() - gpuStart;
    printf("GPU Execution Time: %f sec\n", gpuTime);

    //将数据拷贝回host
    cudaMemcpy(C_host, C_dev, N_C * sizeof(double), cudaMemcpyDeviceToHost);

    //在CPU上计算，并校验数据，计时
    double cpuStart=cpuSecond();
    printf("start gpu calu");
    MatrixMulCPU(C_host, A_host, B_host);
    checkResult(C_dev, C_host, N_C);
    double cpuTime = cpuSecond() - cpuStart;
    printf("CPU Execution Time: %f sec\n", cpuTime);

    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);
    free(A_host);
    free(B_host);
    free(C_host);

    return 0;
}