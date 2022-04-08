#include <iostream>
#include <cuda_runtime.h>

#define CHECK(cmd) \
{\
    cudaError_t error  = cmd;\
    if (error != cudaSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", cudaGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
	  }\
}

using namespace std;

int main() {

  	int count;
	cudaDeviceProp prop;
	CHECK(cudaGetDeviceCount(&count));
	cout <<"当前计算机包含GPU数为"<< count << endl;
	for (int i = 0; i < count; i++){
		CHECK(cudaGetDeviceProperties(&prop, i));
		cout << "当前访问的是第" << i << "块GPU属性" << endl;
		cout << "当前设备名字为: " << prop.name << endl;
		cout << "GPU global mem总量为:" << prop.totalGlobalMem/1024/1024 << " M" << endl;
		cout << "处理器数量：" << prop.multiProcessorCount << endl;
		cout << "每个处理器最大block数量：" << prop.maxBlocksPerMultiProcessor << endl;
		cout << "每个处理器最大线程数：" << prop.maxThreadsPerMultiProcessor << endl;
		cout << "warpSize: " << prop.warpSize << endl;
		cout << "每个gird最大线block：" << prop.maxGridSize[3] << endl;
		cout << "每个block最大线程数：" << prop.maxThreadsPerBlock << endl;
 
	}
}
