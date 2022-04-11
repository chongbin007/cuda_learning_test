#include <stdio.h>
#include <cuda_runtime.h>

#define A_ROW_SIZE 700
#define A_COL_SIZE 3000
#define B_ROW_SIZE 3000
#define B_COL_SIZE 600

#define RAND_SEED 5000

//利用GPU的每个线程计算： A行*B列，得到矩阵C的其中一个元素的值
//当前线程计算的是C[i,j] = A[i]行 *B[j]列
__global__ void matrix_multiplication(double *A, double *B, double *C, size_t N)
{
    size_t offsetThread = (blockIdx.x * blockDim.x + threadIdx.x); //线程号
    size_t totalThreads = gridDim.x * blockDim.x;                  //总线程数

    for (size_t i = offsetThread; i < N; i += totalThreads)
    {
        size_t c_row = i / B_COL_SIZE;
        size_t c_col = i % B_COL_SIZE;
        C[i] = 0;
        //遍历 A的一行，也是B的一列
        for (size_t j = 0; j < A_COL_SIZE; j++)
        {
            size_t indexA = c_row * A_COL_SIZE + j;
            size_t indexB = j * B_COL_SIZE + c_col;
            C[i] += A[indexA] * B[indexB];
        }
    }
}

//这个矩阵其实是个一维数组，打印时我们按照row大小换行
void printMatrix(double *array, int row, int col)
{
    printf("array is: \n");
    int i = 0;
    while (i < row * col)
    {
        printf("%f,", array[i]);
        i++;
        if (i % row == 0)
            printf("\n");
    }
}

int main()
{
    double *d_A, *d_B, *d_C, *h_A, *h_B, *h_C;
    int N_A = A_ROW_SIZE * A_COL_SIZE;
    int N_B = B_ROW_SIZE * B_COL_SIZE;
    int N_C = A_ROW_SIZE * B_COL_SIZE;

    h_A = (double *)malloc(N_A * sizeof(double));
    h_B = (double *)malloc(N_B * sizeof(double));
    h_C = (double *)malloc(N_C * sizeof(double));

    cudaMalloc(&d_A, N_A * sizeof(double));
    cudaMalloc(&d_B, N_B * sizeof(double));
    cudaMalloc(&d_C, N_C * sizeof(double));

    for (int i = 0; i < N_A; i++)
        h_A[i] = rand() % RAND_SEED;

    // printMatrix(h_A,A_ROW_SIZE, A_COL_SIZE);

    for (int i = 0; i < N_B; i++)
        h_B[i] = rand() % RAND_SEED;

    // printMatrix(h_B,B_ROW_SIZE, B_COL_SIZE);

    cudaMemcpy(d_A, h_A, N_A * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N_B * sizeof(double), cudaMemcpyHostToDevice);

    float time_gpu;
    clock_t start_cpu, stop_cpu;
    cudaEvent_t start_GPU, stop_GPU;
    cudaEventCreate(&start_GPU);
    cudaEventCreate(&stop_GPU);
    cudaEventRecord(start_GPU, 0);
    //进入GPU进行计算
    matrix_multiplication<<<600, 700>>>(d_A, d_B, d_C, A_ROW_SIZE * B_COL_SIZE);

    cudaEventRecord(stop_GPU, 0);
    cudaEventSynchronize(start_GPU);
    cudaEventSynchronize(stop_GPU);
    cudaEventElapsedTime(&time_gpu, start_GPU, stop_GPU);
    printf("\nThe time from GPU:\t%f(s)\n", time_gpu / 1000);
    cudaDeviceSynchronize();
    cudaEventDestroy(start_GPU);
    cudaEventDestroy(stop_GPU);

    cudaMemcpy(h_C, d_C, N_C * sizeof(double), cudaMemcpyDeviceToHost);

    printf("\nkernel test:\n");
    // printMatrix(h_C,A_ROW_SIZE, B_COL_SIZE);

    printf("\nwaiting for cpu result ....\n");

    //在host上计算一次矩阵C，C的每个元素都和GPU上计算的进行对比，如果有元素不相等，则挂掉。
    start_cpu = clock();
    double *MatC_check;
    MatC_check = (double *)malloc(N_C * sizeof(double));
    for (int i = 0; i < N_C; i++)
    {
        size_t rowC = i / B_COL_SIZE;
        size_t colC = i % B_COL_SIZE;
        MatC_check[i] = 0;
        for (int index = 0; index < A_COL_SIZE; index++)
            MatC_check[i] += h_A[rowC * A_COL_SIZE + index] * h_B[index * B_COL_SIZE + colC];
        if (fabs(MatC_check[i] - h_C[i]) > 0.001)
        {
            printf("%d %f %f %f\n", i, MatC_check[i], h_C[i], fabs(MatC_check[i] - h_C[i]));
            return 0;
        }
    }
    stop_cpu = clock();
    printf("\nCPU time is %f(s)\n", (float)(stop_cpu - start_cpu) / CLOCKS_PER_SEC);

    printf("\ntest passed!\n");

    //释放已申请的内存
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}

// The time from GPU:      0.098737(s)
// kernel test:
// waiting for cpu result ....
// CPU time is 6.067286(s)