#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
using namespace std;

#define IN_IMG_FILE_NAME "test.yuv"
#define OUT_IMG_FILE_NAME "test_gpu.bmp"
#define OUT_IMG_CPU_FILE_NAME "test_cpu.bmp"
#define IMAGE_SIZE_WIDTH 1920
#define IMAGE_SIZE_HEIGHT 1080
#define IMAGE_SIZE_COEFF 3

typedef unsigned char U8;

typedef struct
{
    unsigned int bfsize;
    unsigned short reserved1;
    unsigned short reserved2;
    unsigned int bfoffBits;
} BTFHeader;

typedef struct
{
    unsigned int bisize;
    int biwidth;
    int biheight;
    unsigned short biplanes;
    unsigned short bibitcount;
    unsigned int bicompression;
    unsigned int bisizeimage;
    int bixpels;
    int biypels;
    unsigned int biclrused;
    unsigned int biclrim;
} BTIHeader;

int readYUVfile(const char *filename, unsigned char *buff, ssize_t size)
{
    int err = 0;
    ifstream in_img(filename, ios::in);
    do
    {
        if (!in_img.is_open())
        {
            cout << "open img file" << filename << "error" << endl;
            err = 1;
            break;
        }
        in_img.read(static_cast<char *>(static_cast<void *>(buff)), size);
        in_img.close();

    } while (0);

    return err;
}

void rbgToBmpFile(const char *filename, unsigned char *buff, const ssize_t size)
{
    BTFHeader file_header = {0};
    BTIHeader info_header = {0};
    unsigned short file_type = 0X4d42;

    file_header.reserved1 = 0;
    file_header.reserved2 = 0;
    file_header.bfsize = 2 + sizeof(short) + sizeof(BTIHeader) + sizeof(BTFHeader) + size * 3;
    file_header.bfoffBits = 0X36;

    info_header.bisize = sizeof(BTIHeader);
    info_header.biwidth = IMAGE_SIZE_WIDTH;
    info_header.biheight = -IMAGE_SIZE_HEIGHT;
    info_header.biplanes = 1;
    info_header.bibitcount = 24;
    info_header.bicompression = 0;
    info_header.bisizeimage = 0;
    info_header.bixpels = 5000;
    info_header.biypels = 5000;
    info_header.biclrused = 0;
    info_header.biclrim = 0;

    ofstream img(filename, ios::out | ios::trunc);
    if (!img.is_open())
    {
        cout << "open bmp file error" << endl;
        exit(0);
    }

    img.write(static_cast<const char *>(static_cast<void *>(&file_type)), sizeof(file_type));
    img.write(static_cast<const char *>(static_cast<void *>(&file_header)), sizeof(file_header));
    img.write(static_cast<const char *>(static_cast<void *>(&info_header)), sizeof(info_header));
    img.write(static_cast<char *>(static_cast<void *>(buff)), size);
    img.close();
}

__global__ void Yuv420ToRgb_gpu(U8 *pYUV, U8 *pRGB, int width, int height)
{
    size_t offsetThread = blockIdx.x * blockDim.x + threadIdx.x; //线程号
    size_t totalThreads = gridDim.x * blockDim.x;                //总线程数

    for (size_t k = offsetThread; k < width * height; k += totalThreads)
    {
        //找到Y、U、V在内存中的首地址
        U8 *pY = pYUV;
        U8 *pU = pYUV + height * width;
        U8 *pV = pU + (height * width / 4);

        U8 *pBGR = NULL;
        U8 R = 0;
        U8 G = 0;
        U8 B = 0;
        U8 Y = 0;
        U8 U = 0;
        U8 V = 0;
        double temp = 0;
        //所在的是第几行第几列的index
        size_t i = k / width;
        size_t j = k % width;

        //找到相应的RGB首地址
        pBGR = pRGB + i * width * 3 + j * 3;

        //取Y、U、V的数据值
        Y = *(pY + i * width + j);
        U = *pU;
        V = *pV;

        // yuv转rgb公式
        temp = Y + ((1.773) * (U - 128));
        B = temp < 0 ? 0 : (temp > 255 ? 255 : (U8)temp);
        temp = (Y - (0.344) * (U - 128) - (0.714) * (V - 128));
        G = temp < 0 ? 0 : (temp > 255 ? 255 : (U8)temp);
        temp = (Y + (1.403) * (V - 128));
        R = temp < 0 ? 0 : (temp > 255 ? 255 : (U8)temp);

        //将转化后的rgb保存在rgb内存中，注意放入的顺序b是最低位
        *pBGR = B;
        *(pBGR + 1) = G;
        *(pBGR + 2) = R;
    }
}

bool Yuv420ToRgb1D(U8 *pYUV, U8 *pRGB, int width, int height)
{
    //找到Y、U、V在内存中的首地址
    U8 *pY = pYUV;
    U8 *pU = pYUV + height * width;
    U8 *pV = pU + (height * width / 4);

    U8 *pBGR = NULL;
    U8 R = 0;
    U8 G = 0;
    U8 B = 0;
    U8 Y = 0;
    U8 U = 0;
    U8 V = 0;
    double temp = 0;
    //矩阵处理，对每个像素点进行转换
    //行
    for (int i = 0; i < height; i++)
    {
        //列
        for (int j = 0; j < width; j++)
        {
            //找到相应的RGB首地址
            pBGR = pRGB + i * width * 3 + j * 3;

            //取Y、U、V的数据值
            Y = *(pY + i * width + j);
            U = *pU;
            V = *pV;

            // yuv转rgb公式
            temp = Y + ((1.773) * (U - 128));
            B = temp < 0 ? 0 : (temp > 255 ? 255 : (U8)temp);
            temp = (Y - (0.344) * (U - 128) - (0.714) * (V - 128));
            G = temp < 0 ? 0 : (temp > 255 ? 255 : (U8)temp);
            temp = (Y + (1.403) * (V - 128));
            R = temp < 0 ? 0 : (temp > 255 ? 255 : (U8)temp);

            //将转化后的rgb保存在rgb内存中，注意放入的顺序b是最低位
            *pBGR = B;
            *(pBGR + 1) = G;
            *(pBGR + 2) = R;
        }
    }
    return true;
}

int main()
{
    U8 *img_yuv = NULL;
    U8 *img_rgb = NULL;
    U8 *d_yuv = NULL;
    U8 *d_rbg = NULL;

    ssize_t size = 0;
    size = IMAGE_SIZE_WIDTH * IMAGE_SIZE_WIDTH * 2;
    // cout << "file size is : IMAGE_SIZE_WIDTH *IMAGE_SIZE_WIDTH * 2 = "<< size <<endl;

    img_yuv = (U8 *)malloc(size);
    img_rgb = (U8 *)malloc(size);

    readYUVfile(IN_IMG_FILE_NAME, img_yuv, size);

    cudaMalloc(&d_yuv, size);
    cudaMalloc(&d_rbg, size);
    cudaMemcpy(d_yuv, img_yuv, size, cudaMemcpyHostToDevice);

    cudaEvent_t start_GPU, stop_GPU;
    float time_gpu;
    cudaEventCreate(&start_GPU);
    cudaEventCreate(&stop_GPU);
    cudaEventRecord(start_GPU, 0);

    Yuv420ToRgb_gpu<<<100, 320>>>(d_yuv, d_rbg, IMAGE_SIZE_WIDTH, IMAGE_SIZE_HEIGHT);

    cudaMemcpy(img_rgb, d_rbg, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_GPU, 0);
    cudaEventSynchronize(start_GPU);
    cudaEventSynchronize(stop_GPU);
    cudaEventElapsedTime(&time_gpu, start_GPU, stop_GPU);
    printf("\nGPU time is %f(ms)\n", time_gpu);
    cudaDeviceSynchronize();

    rbgToBmpFile(OUT_IMG_FILE_NAME, img_rgb, size);

    // run on cpu
    U8 *img_rgb_cpu = NULL;
    img_rgb_cpu = (U8 *)malloc(size);
    clock_t start_cpu, stop_cpu;
    start_cpu = clock();
    Yuv420ToRgb1D(img_yuv, img_rgb_cpu, IMAGE_SIZE_WIDTH, IMAGE_SIZE_HEIGHT);
    stop_cpu = clock();
    printf("\nCPU time is %f(ms)\n", (float)(stop_cpu - start_cpu) / 1000);

    rbgToBmpFile(OUT_IMG_CPU_FILE_NAME, img_rgb_cpu, size);

    cudaEventDestroy(start_GPU);
    cudaEventDestroy(stop_GPU);
    free(img_yuv);
    free(img_rgb);
    cudaFree(d_yuv);
    cudaFree(d_rbg);

    return 0;
}