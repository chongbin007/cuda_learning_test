#include <cuda_runtime.h>
#include <math_functions.h>
#include <iostream>
#include<cmath>
#include<sys/time.h>
#include<stdlib.h>
#include<stdio.h>

#define DEBUG 1
#define thread_per_block 16
#define cuda_CHECK(command){\
    cudaError_t status=command;\
    if(status!=cudaSuccess){\
        std::cerr<<"Error:cuda reports"<<cudaGetErrorString(status)<<endl;\
        std::abort();}}

using namespace std;



//CPU compute  function
void MatrixMultiply_CPU(double *inputA,double *inputB,double *output_r,int M,int N){
    for(int i=0;i<M;i++){
        for(int j=0;j<M;j++){
            double ans=0;
            //cout<<"pos:"<<i<<" "<<j<<endl;
            for(int k=0;k<N;k++){
                double a=inputA[i*N+k];
                double b=inputB[k*M+j];
                //cout<<k<<":"<<a<<" "<<b<<"\t";
                ans+=a*b;
            }
            //cout<<endl;
            output_r[i*M+j]=ans;
        }
    }
}




//GPU compute function
__global__ void MatrixMultiply(double* d_inputA, double* d_inputB, double* d_output,int M,int N)
{
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    int col=blockIdx.y*blockDim.y+threadIdx.y;

    double res=0;
    for(int k=0;k<N;k++){
        double a=d_inputA[row*N+k];
        double b=d_inputB[k*M+col];
        res+=a*b;
    }
    d_output[row*M+col]=res;
}




//GPU shared_memory compute function
__global__ void MatrixMultiply_shared(double* d_inputA, double* d_inputB, double* d_output,int M,int N)
{
    __shared__ double d_inputAs[thread_per_block][thread_per_block];
    __shared__ double d_inputBs[thread_per_block][thread_per_block];

    int bx=blockIdx.x;
    int by=blockIdx.y;
    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int row=bx*blockDim.x+tx;
    int col=by*blockDim.y+ty;

    double res=0;
    for(int m=0;m<N/thread_per_block;++m){
        d_inputAs[tx][ty]=d_inputA[row*N+(m*thread_per_block+ty)];
        d_inputBs[tx][ty]=d_inputB[col+(m*thread_per_block+tx)*M];
        __syncthreads();

        for(int k=0;k<thread_per_block;k++)
        res+=d_inputAs[tx][k]*d_inputBs[k][ty];
        __syncthreads();
    }
    d_output[row*M+col]=res;
}



__global__ void MatrixMultiply_data_prefetch(double* d_inputA, double* d_inputB, double* d_output,int M,int N)
{
    __shared__ double d_inputAs[thread_per_block][thread_per_block];
    __shared__ double d_inputBs[thread_per_block][thread_per_block];

    int bx=blockIdx.x;
    int by=blockIdx.y;
    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int row=bx*blockDim.x+tx;
    int col=by*blockDim.y+ty;

    double res=0;
    int m=0;
    double number1=d_inputA[row*N+(m*thread_per_block+ty)];
    double number2=d_inputB[col+(m*thread_per_block+tx)*M];
    for(m=0;m<N/thread_per_block;++m){
        //d_inputAs[tx][ty]=d_inputA[row*N+(m*thread_per_block+ty)];
        //d_inputBs[tx][ty]=d_inputB[col+(m*thread_per_block+tx)*M];
       d_inputAs[tx][ty]=number1;
        d_inputBs[tx][ty]=number2;
        __syncthreads();

       int j=m+1;
        if(j<N/thread_per_block){
            number1=d_inputA[row*N+(j*thread_per_block+ty)];
            number2=d_inputB[col+(j*thread_per_block+tx)*M];
        }

        for(int k=0;k<thread_per_block;k++)
        res+=d_inputAs[tx][k]*d_inputBs[k][ty];
        __syncthreads();
    }


    d_output[row*M+col]=res;
}





int main(){
    long long int M;
    long long int N;
    struct timeval cstart,gstart,sstart,cend,gend,send;

    M=1024;
    N=512;
    long long int num_input;
    num_input=M*N;
    long long int num_output;
    num_output=M*M;
    double *inputA,*inputB,*output,*output_r,*d_inputA,*d_inputB,*d_output;
    inputA = (double*)malloc(sizeof(double) * num_input);
    inputB = (double*)malloc(sizeof(double) * num_input);
    output = (double*)malloc(sizeof(double) * num_output);
    output_r = (double*)malloc(sizeof(double) * num_output);
    #if DEBUG
    for(int i=0;i<M*N;i++)
            inputA[i]=sin(i);
    for(int i=0;i<M*N;i++)
            inputB[i]=cos(i);
    #endif

    //CPU compute
    gettimeofday(&cstart,NULL);
    MatrixMultiply_CPU(inputA,inputB,output_r,M,N);
    gettimeofday(&cend,NULL);



    // GPU compute
    cuda_CHECK(cudaMalloc((void**)&d_inputA, sizeof(double) * num_input));
    cuda_CHECK(cudaMalloc((void**)&d_inputB, sizeof(double) * num_input));
    cuda_CHECK(cudaMalloc((void**)&d_output, sizeof(double) * num_output)); 
    cuda_CHECK(cudaMemcpy(d_inputA,inputA,sizeof(double) * num_input,cudaMemcpyHostToDevice));
    cuda_CHECK(cudaMemcpy(d_inputB,inputB,sizeof(double) * num_input,cudaMemcpyHostToDevice));
    gettimeofday(&gstart,NULL);
    MatrixMultiply<<<dim3(M/thread_per_block, M/thread_per_block),
                dim3(thread_per_block,thread_per_block),0,0>>>(
                d_inputA, d_inputB, d_output,M,N);

    gettimeofday(&gend,NULL);
    cuda_CHECK(cudaMemcpy(output, d_output, sizeof(double) * num_output, cudaMemcpyDeviceToHost));

    //check
    int flag=0;
    for(int i=0;i<M*M;i++){
        if(fabs(output[i]-output_r[i])>1e-9){
            flag++;
            cout<<"//"<<output[i]<<" "<<output_r[i]<<"//"<<" ";
        }
    }
    if(flag)cout<<"GPU compute:WRONG!!the count of error:"<<flag<<endl;
    else cout<<"GPU compute:PASS!"<<endl;
        double *MatC_check;
    
	MatC_check = (double*)malloc(M*M * sizeof(double));
  

    /*int device=0;
    cudaDeviceProp_t props;
    cudaSetDevice(device);
    cudaGetDeviceProperties(&props,device);

    cout<<"device:"<<props.name<<endl;*/





    //share memory
    gettimeofday(&sstart,NULL);
     MatrixMultiply_shared<<<dim3(M/thread_per_block,M/thread_per_block),
                dim3(thread_per_block,thread_per_block),0,0>>>(
                d_inputA, d_inputB, d_output,M,N);

    gettimeofday(&send,NULL);
    cuda_CHECK(cudaMemcpy(output, d_output, sizeof(double) * num_output, cudaMemcpyDeviceToHost));

    //check
    flag=0;
    for(int i=0;i<M*M;i++){
        if(fabs(output[i]-output_r[i])>1e-9){
            flag++;
            cout<<"//"<<output[i]<<" "<<output_r[i]<<"//"<<" ";
        }
    }
    if(flag)cout<<"GPU shared_memory compute:WRONG!!the count of error:"<<flag<<endl;
    else cout<<"GPU shared_memory compute:PASS!"<<endl;

    
    
    struct timeval dstart,dend;
    //data prefetch
    gettimeofday(&dstart,NULL);
    MatrixMultiply_data_prefetch<<<
                dim3(M/thread_per_block,M/thread_per_block),
                dim3(thread_per_block,thread_per_block),
                0,0>>>
                (d_inputA, d_inputB, d_output,M,N);
    gettimeofday(&dend,NULL);
    cuda_CHECK(cudaMemcpy(output, d_output, sizeof(double) * num_output, cudaMemcpyDeviceToHost));

    //check
    flag=0;
    for(int i=0;i<M*M;i++){
        if(fabs(output[i]-output_r[i])>1e-9){
            flag++;
            cout<<"//"<<output[i]<<" "<<output_r[i]<<"//"<<" ";
        }
    }
    if(flag)cout<<"GPU with shared_memory and data prefetch compute:WRONG!!the count of error:"<<flag<<endl;
    else cout<<"GPU with shared_memory and data prefetch compute:PASS!"<<endl;



    //compute time
    double ctimeuse=1000000*(cend.tv_sec-cstart.tv_sec)+cend.tv_usec-cstart.tv_usec;
    double gtimeuse=1000000*(gend.tv_sec-gstart.tv_sec)+gend.tv_usec-gstart.tv_usec;
    double stimeuse=1000000*(send.tv_sec-sstart.tv_sec)+send.tv_usec-sstart.tv_usec;
    double dtimeuse=1000000*(dend.tv_sec-dstart.tv_sec)+dend.tv_usec-dstart.tv_usec;
    cout<<"time of CPU is:"<<ctimeuse/1000<<"ms"<<endl;
    cout<<"time of GPU is:"<<gtimeuse/1000<<"ms"<<endl;
    cout<<"time of GPU with shared_memory is:"<<stimeuse/1000<<"ms"<<endl;
    cout<<"time of GPU with shared_memory and data prefetch is:"<<dtimeuse/1000<<"ms"<<endl;


    //free
    free(inputA);
    free(inputB);
    free(output);
    free(output_r);
    cuda_CHECK(cudaFree(d_inputA));
    cuda_CHECK(cudaFree(d_inputB));
    cuda_CHECK(cudaFree(d_output));
    return 0;
}