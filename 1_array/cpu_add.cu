#include <iostream>
#include <math.h>
#include <time.h>
#include <sys/time.h>

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

// function to add the elements of two arrays
void add(int n, float *x, float *y)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

int main(void)
{
    int N = 10000000; // 1M elements

    float *x = new float[N];
    float *y = new float[N];

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    double gpuStart = cpuSecond();

    // Run kernel on 1M elements on the CPU
    add(N, x, y);
    double gpuTime = cpuSecond() - gpuStart;
    printf("Execution Time: %f ms\n", gpuTime * 1000);

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    delete[] x;
    delete[] y;

    return 0;
}
//26ms