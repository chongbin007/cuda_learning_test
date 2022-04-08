

#include <iostream>
#include <fstream>

using namespace std;

#define IN_IMG_FILE_NAME    "3.yuv"
#define OUT_IMG_FILE_NAME   "test.bmp"
#define IMAGE_SIZE_WIDTH    1920
#define IMAGE_SIZE_HEIGHT   1080
#define IMAGE_SIZE_COEFF    3

typedef unsigned char U8;
typedef struct
{
    unsigned int bfsize;
    unsigned short reserved1;
    unsigned short reserved2;
    unsigned int bfoffBits;
} Bitmapfileheader;

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
} Bitmapinfoheader;



int readImgFile(const char *filename, unsigned char *buff, ssize_t size)
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

        in_img.read(static_cast<char*>(static_cast<void*>(buff)), size);
        in_img.close();

    } while (0);

    return err;
}


bool Yuv420ToRgb2D(U8* pYUV, U8* pRGB, int width, int height)
{
    //找到Y、U、V在内存中的首地址
    U8* pY = pYUV;
    U8* pU = pYUV + height*width;
    U8* pV = pU + (height*width / 4);

    U8* pBGR = NULL;
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
            pBGR = pRGB + i*width * 3 + j * 3;

            //取Y、U、V的数据值
            Y = *(pY + i*width + j);
            U = *pU;
            V = *pV;

            //yuv转rgb公式
            temp = Y + ((1.773) * (U - 128));
            B = temp < 0 ? 0 : (temp>255 ? 255 : (U8)temp);
            temp = (Y - (0.344) * (U - 128) - (0.714) * (V - 128));
            G = temp < 0 ? 0 : (temp>255 ? 255 : (U8)temp);
            temp = (Y + (1.403)*(V - 128));
            R = temp < 0 ? 0 : (temp>255 ? 255 : (U8)temp);

            //将转化后的rgb保存在rgb内存中，注意放入的顺序b是最低位
            *pBGR = B;
            *(pBGR + 1) = G;
            *(pBGR + 2) = R;

            if (j % 2 != 0) {
                *pU++;
                *pV++;
            }
        }
        if (i % 2 == 0) {
            pU = pU - width / 2;
            pV = pV - width / 2;
        }
    }
    return true;
}


bool Yuv420ToRgb1D(U8* pYUV, U8* pRGB, int width, int height)
{
    //找到Y、U、V在内存中的首地址
    U8* pY = pYUV;
    U8* pU = pYUV + height*width;
    U8* pV = pU + (height*width / 4);

    U8* pBGR = NULL;
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
            pBGR = pRGB + i*width * 3 + j * 3;

            //取Y、U、V的数据值
            Y = *(pY + i*width + j);
            U = *pU;
            V = *pV;

            //yuv转rgb公式
            temp = Y + ((1.773) * (U - 128));
            B = temp < 0 ? 0 : (temp>255 ? 255 : (U8)temp);
            temp = (Y - (0.344) * (U - 128) - (0.714) * (V - 128));
            G = temp < 0 ? 0 : (temp>255 ? 255 : (U8)temp);
            temp = (Y + (1.403)*(V - 128));
            R = temp < 0 ? 0 : (temp>255 ? 255 : (U8)temp);

            //将转化后的rgb保存在rgb内存中，注意放入的顺序b是最低位
            *pBGR = B;
            *(pBGR + 1) = G;
            *(pBGR + 2) = R;

            if (j % 2 != 0) {
                *pU++;
                *pV++;
            }
        }
        if (i % 2 == 0) {
            pU = pU - width / 2;
            pV = pV - width / 2;
        }
    }
    return true;
}


void saveBmpFile(const char *filename, unsigned char *buff, const ssize_t size)
{
    Bitmapfileheader file_header = { 0 };
    Bitmapinfoheader info_header = { 0 };
    unsigned short file_type = 0X4d42;

    file_header.reserved1 = 0;
    file_header.reserved2 = 0;
    file_header.bfsize = 2 + sizeof(short) + sizeof(Bitmapinfoheader) + sizeof(Bitmapfileheader) + size*3;
    file_header.bfoffBits = 0X36;

    info_header.bisize = sizeof(Bitmapinfoheader);
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
        cout << "open bmp file error" <<endl;
        exit(0);
    }

    img.write(static_cast<const char*>(static_cast<void*>(&file_type)), sizeof(file_type));
    img.write(static_cast<const char*>(static_cast<void*>(&file_header)), sizeof(file_header));
    img.write(static_cast<const char*>(static_cast<void*>(&info_header)), sizeof(info_header));
    img.write(static_cast<char*>(static_cast<void*>(buff)), size);
    img.close();
}

int main()
{
    int err = 0;
    ssize_t size = 0;
    unsigned char *img_yuv = NULL;
    unsigned char *img_rgb = NULL;

    size = IMAGE_SIZE_WIDTH *IMAGE_SIZE_WIDTH * 2;

    cout << "file size is : IMAGE_SIZE_WIDTH *IMAGE_SIZE_WIDTH * 2 = "<< size <<endl;

    img_yuv = (unsigned char*)malloc(size);
    img_rgb = (unsigned char*)malloc(size);

    err = readImgFile(IN_IMG_FILE_NAME, img_yuv, size);
    if (0 != err)
    {
        cout << "read img file error" <<endl;
        exit(0);
    }

    Yuv420ToRgb1D(img_yuv, img_rgb, IMAGE_SIZE_WIDTH, IMAGE_SIZE_HEIGHT);

    saveBmpFile(OUT_IMG_FILE_NAME, img_rgb, size);

    return 0;
}