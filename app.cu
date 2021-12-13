#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
// #include "cuda_runtime.h"
// #include "device_launch_parameters.h"

using namespace std;

// #define f_idx(a, c, x, y, A, C, X, Y) ((x) + (X) * ((y) + (Y) * ((c) + (C) * (a))))

#define ReLU(v) (max((v), 0.0f))

#define idx(ch, row, col, size) ((ch)*(size)*(size) + (row)*(size) + col)
#define f_idx(inpCh, outCh, InpCh) ((inpCh) + (InpCh*outCh))

void printVector2d(vector<float> vec, int size, int channel) {
   printf("Channel %d\n", channel);
   for (int y = 0; y < size; y++) {
      for (int x = 0; x < size; x++)
         printf("%f ", vec[idx(channel, y, x, size)]);
      printf("\n");
   }
   printf("\n");
}

void printFilter2(vector<float> f, int inputChannels, int outputChannels) {
   printf("Filter 2:\n");
   for (int o = 0; o < outputChannels; o++) {
      for (int i = 0; i < inputChannels; i++)
         printf("%f ", f[f_idx(i, o, inputChannels)]);
      printf("\n");
   }
   printf("\n");
}

__shared__ float buffer[];
__global__ void depthwise_separable_convolution(
      int inputSize, int inputChannels,
      int outputChannels, const float* input, float* output, 
      const float* filter1, const float* filter2, int filterSize
) {
   const int x = blockIdx.x;
   const int y = blockIdx.y;
   const int ch = threadIdx.x;
   float t;

   if (ch < inputChannels) {
      t = 0;
      for (int f1y = 0; f1y < filterSize; f1y++){
         for (int f1x = 0; f1x < filterSize; f1x++) {
            t += input[idx(ch, y + f1y, x + f1x, inputSize+2)] * filter1[idx(ch, f1y, f1x, filterSize)];
         }
      }
      buffer[ch] = t;
   }   

   __syncthreads();

   if (ch < outputChannels) {
      t = 0;
      for (int inpCh = 0; inpCh < inputChannels; inpCh++) {
         t += buffer[ch] * filter2[f_idx(inpCh, ch, inputChannels)];
      }
      output[idx(ch, y, x, inputSize)] = ReLU(t);
   }   
}


__shared__ float tBuffer[];
__global__ void depthwise_separable_convolution_toeplitz(
      int inputSize, int inputChannels,
      int outputChannels, const float* input, float* output, 
      const float* filter1, const float* filter2, int filterSize
) {
   const int x = blockIdx.x;
   const int y = blockIdx.y;
   const int ch = threadIdx.x;
   float t;

   if (ch < inputChannels) {
      // int toeplitzRowIdx = 
      // int groupSizeX = inputSize+2;
      int groupSizeY = inputSize+2 - filterSize + 1;
      int startGroupIdx = (y*inputSize + x) / groupSizeY;
      int zeroesLeft = (y*inputSize + x) % groupSizeY;

      t = 0;
      for (int fRow = 0; fRow < filterSize; fRow++) {
         int inpRow = startGroupIdx + fRow;
         for (int fCol = 0; fCol < filterSize; fCol++) {            
            // int inpCol = ;
            t += filter1[idx(ch, fRow, fCol, filterSize)] * input[idx(ch, inpRow, zeroesLeft + fCol, inputSize+2)];
         }
      }
      tBuffer[ch] = t;
   }   

   __syncthreads();

   if (ch < outputChannels) {
      t = 0;
      for (int inpCh = 0; inpCh < inputChannels; inpCh++) {
         t += tBuffer[ch] * filter2[f_idx(inpCh, ch, inputChannels)];
      }
      output[idx(ch, y, x, inputSize)] = ReLU(t);
   }   
}


bool check_error_status(cudaError_t status, const char *error_message) {
   if (status != cudaSuccess) {
      fprintf(stderr, error_message);
      return true;
   }
   return false;
}

bool test(int inputSize, int inputChannels, int outputChannels, int filterSize,
      vector<float> &hInput, vector<float> &hOutput, vector<float> &hFilter1, vector<float> &hFilter2
      // , float &funcTime, float &memTime
) {   
   cudaError_t status; 
   // cudaEvent_t funcStart, funcStop, memStart, memStop;
   // cudaEventCreate(&funcStart); 
   // cudaEventCreate(&funcStop);
   // cudaEventCreate(&memStart);
   // cudaEventCreate(&memStop);

   status = cudaSetDevice(0);
   if (check_error_status(status, "cudaSetDevice fail\n")) return false;

   // cudaEventRecord(memStart);
   float *dInput, *dOutput, *dFilter1, *dFilter2;
   status = cudaMalloc((void**)&dInput, hInput.size()*sizeof(float));
   if (check_error_status(status, "cudaMalloc fail\n"))
      return false;
   status = cudaMalloc((void**)&dOutput, hOutput.size()*sizeof(float));
   if (check_error_status(status, "cudaMalloc fail\n"))
      return false;
   status = cudaMalloc((void**)&dFilter1, hFilter1.size()*sizeof(float));
   if (check_error_status(status, "cudaMalloc fail\n"))
      return false;
   status = cudaMalloc((void**)&dFilter2, hFilter2.size()*sizeof(float));
   if (check_error_status(status, "cudaMalloc fail\n"))
      return false;

   status = cudaMemcpy(dInput, hInput.data(), hInput.size()*sizeof(float), cudaMemcpyHostToDevice);
   if (check_error_status(status, "cudaMemCpy fail\n"))
      return false;
   status = cudaMemcpy(dFilter1, hFilter1.data(), hFilter1.size()*sizeof(float), cudaMemcpyHostToDevice);
   if (check_error_status(status, "cudaMemCpy fail\n"))
      return false;
   status = cudaMemcpy(dFilter2, hFilter2.data(), hFilter2.size()*sizeof(float), cudaMemcpyHostToDevice);
   if (check_error_status(status, "cudaMemCpy fail\n"))
      return false;

   // cudaEventRecord(memStop);
   // cudaEventSynchronize(memStop);
   // float temp;
   // cudaEventElapsedTime(&temp, memStart, memStop);
   // memTime += temp;

   dim3 dimBlock(max(inputChannels, outputChannels), 1);
   dim3 dimGrid(inputSize, inputSize);
   depthwise_separable_convolution<<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(
      inputSize, inputChannels,
      outputChannels, dInput, dOutput, 
      dFilter1, dFilter2, filterSize
   );

   // cudaEventRecord(funcStop);
   // cudaEventSynchronize(funcStop);
   // cudaEventElapsedTime(&temp, funcStart, funcStop);
   // funcTime += temp;

   // cudaEventRecord(memStart);
   status = cudaMemcpy(hOutput.data(), dOutput, hOutput.size()*sizeof(float), cudaMemcpyDeviceToHost);
   if (check_error_status(status, "couldn't load device output to host"))
      return false;
   // cudaEventRecord(memStop);
   // cudaEventSynchronize(memStop);
   // cudaEventElapsedTime(&temp, memStart, memStop);
   // memTime += temp;

   cudaFree(dInput);
   cudaFree(dOutput);
   cudaFree(dFilter1);
   cudaFree(dFilter2);
   // cudaEventDestroy(funcStart);
   // cudaEventDestroy(funcStop);
   // cudaEventDestroy(memStart);
   // cudaEventDestroy(memStop);
   return true;
}

bool test_toeplitz(int inputSize, int inputChannels, int outputChannels, int filterSize,
      vector<float> &hInput, vector<float> &hOutput, vector<float> &hFilter1, vector<float> &hFilter2
      // , float &funcTime, float &memTime
) {
   cudaError_t status;
   // cudaEvent_t funcStart, funcStop, memStart, memStop;
   // cudaEventCreate(&funcStart); 
   // cudaEventCreate(&funcStop);
   // cudaEventCreate(&memStart);
   // cudaEventCreate(&memStop);

   

   status = cudaSetDevice(0);
   if (check_error_status(status, "cudaSetDevice fail\n")) return false;

   // cudaEventRecord(memStart);
   float *dInput, *dOutput, *dFilter1, *dFilter2;
   status = cudaMalloc((void**)&dInput, hInput.size()*sizeof(float));
   if (check_error_status(status, "cudaMalloc fail\n"))
      return false;
   status = cudaMalloc((void**)&dOutput, hOutput.size()*sizeof(float));
   if (check_error_status(status, "cudaMalloc fail\n"))
      return false;
   status = cudaMalloc((void**)&dFilter1, hFilter1.size()*sizeof(float));
   if (check_error_status(status, "cudaMalloc fail\n"))
      return false;
   status = cudaMalloc((void**)&dFilter2, hFilter2.size()*sizeof(float));
   if (check_error_status(status, "cudaMalloc fail\n"))
      return false;

   status = cudaMemcpy(dInput, hInput.data(), hInput.size()*sizeof(float), cudaMemcpyHostToDevice);
   if (check_error_status(status, "cudaMemCpy fail\n"))
      return false;
   status = cudaMemcpy(dFilter1, hFilter1.data(), hFilter1.size()*sizeof(float), cudaMemcpyHostToDevice);
   if (check_error_status(status, "cudaMemCpy fail\n"))
      return false;
   status = cudaMemcpy(dFilter2, hFilter2.data(), hFilter2.size()*sizeof(float), cudaMemcpyHostToDevice);
   if (check_error_status(status, "cudaMemCpy fail\n"))
      return false;
   
   // cudaEventRecord(memStop);
   // cudaEventSynchronize(memStop);
   // float temp;
   // cudaEventElapsedTime(&temp, memStart, memStop);
   // memTime += temp;

   // cudaEventRecord(funcStart);
   dim3 dimBlock(max(inputChannels, outputChannels), 1);
   dim3 dimGrid(inputSize, inputSize);
   depthwise_separable_convolution_toeplitz<<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(
      inputSize, inputChannels,
      outputChannels, dInput, dOutput, 
      dFilter1, dFilter2, filterSize
   );

   // cudaEventRecord(funcStop);
   // cudaEventSynchronize(funcStop);
   // cudaEventElapsedTime(&temp, funcStart, funcStop);
   // funcTime += temp;

   // cudaEventRecord(memStart);
   status = cudaMemcpy(hOutput.data(), dOutput, hOutput.size()*sizeof(float), cudaMemcpyDeviceToHost);
   if (check_error_status(status, "couldn't load device output to host"))
      return false;
   // cudaEventRecord(memStop);
   // cudaEventSynchronize(memStop);
   // cudaEventElapsedTime(&temp, memStart, memStop);
   // memTime += temp;

   cudaFree(dInput);
   cudaFree(dOutput);
   cudaFree(dFilter1);
   cudaFree(dFilter2);
   // cudaEventDestroy(funcStart);
   // cudaEventDestroy(funcStop);
   // cudaEventDestroy(memStart);
   // cudaEventDestroy(memStop);
   return true;
}


bool fillInput(vector<float> &inp, int channelNum, int inpSize) {
   if (inp.size() < channelNum*(inpSize+2)*(inpSize+2)){
      printf("Couldn't fill input vector\n");
      return false;
   }
   for (int ch = 0; ch < channelNum; ch++) {
      for (int y = 0; y < inpSize+2; y++)
         for (int x = 0; x < inpSize+2; x++)
            inp[idx(ch, y, x, inpSize+2)] = rand() % 3 + 1.0 / (1.0 + rand() % 3);
      for (int x = 0, y = 0; x < inpSize+2; x++)
         inp[idx(ch, y, x, inpSize+2)] = 0;
      for (int x = 0, y = inpSize+1; x < inpSize+2; x++)
         inp[idx(ch, y, x, inpSize+2)] = 0;   

      for (int x = 0, y = 0; y < inpSize+2; y++)
         inp[idx(ch, y, x, inpSize+2)] = 0;
      for (int x = inpSize+1, y = 0; y < inpSize+2; y++)
         inp[idx(ch, y, x, inpSize+2)] = 0;   
   }  
   return true;
}

bool fillFilter1(vector<float> &f, int channelNum, int dimSize) {
   if (f.size() < channelNum*dimSize*dimSize) {
      printf("Couldn't fill filter 1\n");
      return false;
   }
   for (int ch = 0; ch < channelNum; ch++) {
      for (int y = 0; y < dimSize; y++)
         for (int x = 0; x < dimSize; x++)
            f[idx(ch, y, x, dimSize)] = rand() % 3 + 1.0 / (1.0 + rand() % 3);
   }
   return true;   
}

bool fillFilter2(vector<float> &f, int inputChannels, int outputChannels) {
   if (f.size() < inputChannels*outputChannels) {
      printf("Couldn't fill filter 2\n");
      return false;
   }
   for (int oChannel = 0; oChannel < outputChannels; oChannel++)
      for (int iChannel = 0; iChannel < inputChannels; iChannel++)
         f[f_idx(iChannel, oChannel, inputChannels)] = rand() % 3 + 1.0 / (1.0 + rand() % 3);
   return true;
}


bool mini_test() {
   const int inputSize = 4;
	const int inputChannels = 2;
	const int outputChannels = 2;
	const int filterSize = 3;
   cudaError_t status; 
   printf("Mini test:\n"); 

   status = cudaSetDevice(0);
   if (check_error_status(status, "cudaSetDevice fail\n")) return false;
      
   vector<float> hInput(inputChannels*(inputSize+2)*(inputSize+2));
   vector<float> hOutput(outputChannels*inputSize*inputSize);
   vector<float> hFilter1(inputChannels*filterSize*filterSize);
   vector<float> hFilter2(inputChannels*outputChannels);
   fillInput(hInput, inputChannels, inputSize);
   printf("Input:\n");
   printVector2d(hInput, inputSize+2, 0);
   // printVector2d(hInput, inputSize+2, 1);

   fillFilter1(hFilter1, inputChannels, filterSize);
   printf("Filter 1:\n");
   printVector2d(hFilter1, filterSize, 0);
   // printVector2d(hFilter1, filterSize, 1);

   fillFilter2(hFilter2, inputChannels, outputChannels);
   printFilter2(hFilter2, inputChannels, outputChannels);

   float *dInput, *dOutput, *dFilter1, *dFilter2;
   status = cudaMalloc((void**)&dInput, hInput.size()*sizeof(float));
   if (check_error_status(status, "cudaMalloc fail\n"))
      return false;
   status = cudaMalloc((void**)&dOutput, hOutput.size()*sizeof(float));
   if (check_error_status(status, "cudaMalloc fail\n"))
      return false;
   status = cudaMalloc((void**)&dFilter1, hFilter1.size()*sizeof(float));
   if (check_error_status(status, "cudaMalloc fail\n"))
      return false;
   status = cudaMalloc((void**)&dFilter2, hFilter2.size()*sizeof(float));
   if (check_error_status(status, "cudaMalloc fail\n"))
      return false;
   // printf("cuda malloc done\n");

   status = cudaMemcpy(dInput, hInput.data(), hInput.size()*sizeof(float), cudaMemcpyHostToDevice);
   if (check_error_status(status, "cudaMemCpy fail\n"))
      return false;
   status = cudaMemcpy(dFilter1, hFilter1.data(), hFilter1.size()*sizeof(float), cudaMemcpyHostToDevice);
   if (check_error_status(status, "cudaMemCpy fail\n"))
      return false;
   status = cudaMemcpy(dFilter2, hFilter2.data(), hFilter2.size()*sizeof(float), cudaMemcpyHostToDevice);
   if (check_error_status(status, "cudaMemCpy fail\n"))
      return false;
   // printf("cuda memcpy done\n");

   dim3 dimBlock(max(inputChannels, outputChannels), 1);
   dim3 dimGrid(inputSize, inputSize);
   // printf("func started\n");
   depthwise_separable_convolution<<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(
      inputSize, inputChannels,
      outputChannels, dInput, dOutput, 
      dFilter1, dFilter2, filterSize
   );
   // printf("func ended\n");

   status = cudaMemcpy(hOutput.data(), dOutput, hOutput.size()*sizeof(float), cudaMemcpyDeviceToHost);
   if (check_error_status(status, "couldn't load device output to host"))
      return false;
   printf("Output standard:\n");
   printVector2d(hOutput, inputSize, 0);
   // printVector2d(hOutput, inputSize, 1);

   // printf("toeplitz started\n");
   depthwise_separable_convolution_toeplitz<<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(
      inputSize, inputChannels,
      outputChannels, dInput, dOutput, 
      dFilter1, dFilter2, filterSize
   );
   // printf("toeplitz ended\n");

   status = cudaMemcpy(hOutput.data(), dOutput, hOutput.size()*sizeof(float), cudaMemcpyDeviceToHost);
   if (check_error_status(status, "couldn't load device output to host"))
      return false;
   printf("Output toeplitz:\n");
   printVector2d(hOutput, inputSize, 0);
   // printVector2d(hOutput, inputSize, 1);

   cudaFree(dInput);
   cudaFree(dOutput);
   cudaFree(dFilter1);
   cudaFree(dFilter2);
   return true;
}

void test1() {
   // const int inputChannels = 3;
	// const int outputChannelsArr[]{3, 8, 16, 32, 64, 128, 256};
	// const int filterSize = 3;
   // // mini_test();   
   // const int testNum = 10;
   // const int inputSizeMax = 1 << 10;
   // ofstream res("test_res.txt");   
   // res << "test num:" << testNum << endl;

   // for (int outputChannels : outputChannelsArr) {
   //    for (int inputSize = 256; inputSize <= inputSizeMax; inputSize <<= 1) {
   //       vector<float> hInput(inputChannels*(inputSize+2)*(inputSize+2));
   //       vector<float> hOutput(outputChannels*inputSize*inputSize);
   //       vector<float> hOutputToeplitz(outputChannels*inputSize*inputSize);
   //       vector<float> hFilter1(inputChannels*filterSize*filterSize);
   //       vector<float> hFilter2(inputChannels*outputChannels);

   //       res << "input:output(size*size) " << inputChannels << ":" << outputChannels << "(" 
   //          << inputSize << "*" << inputSize << ")" << endl;
   //       printf("input:output(size*size) %d:%d(%d*%d)\n\t", inputChannels, outputChannels, inputSize, inputSize);
         

   //       float funcTime = 0, memTime = 0;
   //       for (int i = 1; i <= testNum; i++) {
   //          fillInput(hInput, inputChannels, inputSize);
   //          fillFilter1(hFilter1, inputChannels, filterSize);
   //          fillFilter2(hFilter2, inputChannels, outputChannels);
   //          test(inputSize, inputChannels, outputChannels, filterSize,
   //             hInput, hOutput, hFilter1, hFilter2, 
   //             funcTime, memTime); 
   //          // test_toeplitz(inputSize, inputChannels, outputChannels, filterSize,
   //          //    hInput, hOutputToeplitz, hFilter1, hFilter2, 
   //          //    funcTime, memTime);
   //          printf("%d.pass  ", i);
   //          if (i% 10 == 0)
   //             printf("\n");
   //       }
   //       res << "   function: " << funcTime << "ms; " << "memory: " << memTime << "ms" << endl;
   //       res << "   function avg: " << funcTime / testNum << "ms; " << "memory avg: " << memTime / testNum << "ms" << endl; 
   //    }  
   //    res << endl;
   // }
   // res.close();
}

void test2() {
   const int inputChannels = 3;
   const int outputChannels = 16;
   const int inputSize = 256;
   const int filterSize = 3;
   const int testNum = 100;

   vector<float> hInput(inputChannels*(inputSize+2)*(inputSize+2));
   vector<float> hOutput(outputChannels*inputSize*inputSize);
   vector<float> hOutputToeplitz(outputChannels*inputSize*inputSize);
   vector<float> hFilter1(inputChannels*filterSize*filterSize);
   vector<float> hFilter2(inputChannels*outputChannels);


   for (int i = 1; i <= testNum; i++) {
      fillInput(hInput, inputChannels, inputSize);
      fillFilter1(hFilter1, inputChannels, filterSize);
      fillFilter2(hFilter2, inputChannels, outputChannels);
      test(inputSize, inputChannels, outputChannels, filterSize,
         hInput, hOutput, hFilter1, hFilter2
      );
      test_toeplitz(inputSize, inputChannels, outputChannels, filterSize,
         hInput, hOutputToeplitz, hFilter1, hFilter2
      );
      int size = hOutput.size();
      for (int idx = 0; idx < size; idx++) {
         if (hOutput[idx] != hOutputToeplitz[idx]) {
            printf("%d. fail\n", i);
            printf("Standart output: %f;  Toeplitz output:%f;\n", hOutput[idx], hOutputToeplitz[idx]);
            printf("idx=%d", idx);
         }
      }

      printf("%d. pass  ", i);
      if (i % 10 == 0) {
         printf("\n");
      }
   }
}

void unetx1() {
   // depthwise_conv2d_68 - conv2d_115
   // conv_1_1
   const int inputSize = 256;
   const int filterSize = 3;
   const int inputChannels = 3;
   const int outputChannels = 16;
   const int testNum = 1000;
   printf("UNET X: depthwise_conv2d_68 - const2d_115\n");
   printf("input:output(size*size) %d:%d(%d*%d)\n\t", inputChannels, outputChannels, inputSize, inputSize);

   vector<float> hInput(inputChannels*(inputSize+2)*(inputSize+2));
   vector<float> hOutput(outputChannels*inputSize*inputSize);
   vector<float> hOutputToeplitz(outputChannels*inputSize*inputSize);
   vector<float> hFilter1(inputChannels*filterSize*filterSize);
   vector<float> hFilter2(inputChannels*outputChannels);

   for (int i = 1; i <= testNum; i++) {
      fillInput(hInput, inputChannels, inputSize);
      fillFilter1(hFilter1, inputChannels, filterSize);
      fillFilter2(hFilter2, inputChannels, outputChannels);
      test(inputSize, inputChannels, outputChannels, filterSize,
         hInput, hOutput, hFilter1, hFilter2
      );
      test_toeplitz(inputSize, inputChannels, outputChannels, filterSize,
         hInput, hOutputToeplitz, hFilter1, hFilter2
      );
      int size = hOutput.size();
      for (int idx = 0; idx < size; idx++) {
         if (hOutput[idx] != hOutputToeplitz[idx]) {
            printf("%d. fail\n", i);
            printf("Standart output: %f;  Toeplitz output:%f;\n", hOutput[idx], hOutputToeplitz[idx]);
            printf("idx=%d", idx);
         }
      }
      // printf("%d. pass  ", i);
      // if (i % 10 == 0) {
      //    printf("\n");
      // }
      if (i % 100 == 0) {
         printf("%d.pass  ", i);
      }
   }
}

void unetx2() {
   // depthwise_conv2d_72 - conv2d_119
   // conv_3_1

   const int inputSize = 64;
   const int filterSize = 3;
   const int inputChannels = 3;
   const int outputChannels = 32;
   const int testNum = 1000;
   printf("UNET X: depthwise_conv2d_72 - conv2d_119\n");
   printf("input:output(size*size) %d:%d(%d*%d)\n\t", inputChannels, outputChannels, inputSize, inputSize);

   vector<float> hInput(inputChannels*(inputSize+2)*(inputSize+2));
   vector<float> hOutput(outputChannels*inputSize*inputSize);
   vector<float> hOutputToeplitz(outputChannels*inputSize*inputSize);
   vector<float> hFilter1(inputChannels*filterSize*filterSize);
   vector<float> hFilter2(inputChannels*outputChannels);

   for (int i = 1; i <= testNum; i++) {
      fillInput(hInput, inputChannels, inputSize);
      fillFilter1(hFilter1, inputChannels, filterSize);
      fillFilter2(hFilter2, inputChannels, outputChannels);
      test(inputSize, inputChannels, outputChannels, filterSize,
         hInput, hOutput, hFilter1, hFilter2
      );
      test_toeplitz(inputSize, inputChannels, outputChannels, filterSize,
         hInput, hOutputToeplitz, hFilter1, hFilter2
      );
      int size = hOutput.size();
      for (int idx = 0; idx < size; idx++) {
         if (hOutput[idx] != hOutputToeplitz[idx]) {
            printf("%d. fail\n", i);
            printf("Standart output: %f;  Toeplitz output:%f;\n", hOutput[idx], hOutputToeplitz[idx]);
            printf("idx=%d", idx);
         }
      }
      // printf("%d. pass  ", i);
      // if (i % 10 == 0) {
      //    printf("\n");
      // }
      if (i % 100 == 0) {
         printf("%d.pass  ", i);
      }
   }
}

void unetx3() {
   // depthwise_conv2d_77 - conv2d_124
   // conv_5_2


   const int inputSize = 16;
   const int filterSize = 3;
   const int inputChannels = 3;
   const int outputChannels = 256;
   const int testNum = 1000;
   printf("UNET X: depthwise_conv2d_77 - conv2d_124\n");
   printf("input:output(size*size) %d:%d(%d*%d)\n\t", inputChannels, outputChannels, inputSize, inputSize);

   vector<float> hInput(inputChannels*(inputSize+2)*(inputSize+2));
   vector<float> hOutput(outputChannels*inputSize*inputSize);
   vector<float> hOutputToeplitz(outputChannels*inputSize*inputSize);
   vector<float> hFilter1(inputChannels*filterSize*filterSize);
   vector<float> hFilter2(inputChannels*outputChannels);

   for (int i = 1; i <= testNum; i++) {
      fillInput(hInput, inputChannels, inputSize);
      fillFilter1(hFilter1, inputChannels, filterSize);
      fillFilter2(hFilter2, inputChannels, outputChannels);
      test(inputSize, inputChannels, outputChannels, filterSize,
         hInput, hOutput, hFilter1, hFilter2
      );
      test_toeplitz(inputSize, inputChannels, outputChannels, filterSize,
         hInput, hOutputToeplitz, hFilter1, hFilter2
      );
      int size = hOutput.size();
      for (int idx = 0; idx < size; idx++) {
         if (hOutput[idx] != hOutputToeplitz[idx]) {
            printf("%d. fail\n", i);
            printf("Standart output: %f;  Toeplitz output:%f;\n", hOutput[idx], hOutputToeplitz[idx]);
            printf("idx=%d", idx);
         }
      }
      // printf("%d. pass  ", i);
      // if (i % 10 == 0) {
      //    printf("\n");
      // }
      if (i % 100 == 0) {
         printf("%d.pass  ", i);
      }
   }
}

int main() {
	// test1();
   // test2();
   unetx1();
   // unetx2();
   // unetx3();
}