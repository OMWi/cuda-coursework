#include <iostream>
#include <vector>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

#define idx(c, x, y, C, X, Y) ((x) + (X) * ((y) + (Y) * (c)))
// #define f_idx(a, c, x, y, A, C, X, Y) ((x) + (X) * ((y) + (Y) * ((c) + (C) * (a))))
#define f_idx(inpCh, outCh, InpCh) ((inpCh) + (InpCh*outCh))
#define ReLU(v) (max((v), 0.0f))

#define idx2(ch, row, col, size) ((ch)*(size)*(size) + (row)*(size) + col)


__shared__ float buffer[];
__global__ void depthwise_separable_convolution(
      int inputSize, int inputChannels,
      int outputChannels, const float* input, float* output, 
      const float* filter1, const float* filter2, int filterSize
) {
   const int n2y = blockIdx.x;
   const int n2x = blockIdx.y;
   const int ch = threadIdx.x;

   if (ch > inputChannels) return;
   float t = 0;
   for (int f1y = 0; f1y < filterSize; f1y++){
      for (int f1x = 0; f1x < filterSize; f1x++) {
         t += input[idx(ch, n2x + f1x, n2y + f1y, inputChannels, inputSize, inputSize)] * filter1[idx(ch, f1x, f1y, inputChannels, filterSize, filterSize)];
      }
   }
   buffer[ch] = t;

   __syncthreads();

   if (ch >= outputChannels) return;
   t = 0;
   for (int ch2 = 0; ch2 < inputChannels; ch2++) {
      t += buffer[ch2] * filter2[f_idx(ch, ch2, inputChannels)];
   }
   output[idx(ch, n2x, n2y, outputChannels, inputSize, inputSize)] = ReLU(t);
}

__global__ void depthwise_separable_convolution_toeplitz(
      int inputSize, int inputChannels,
      int outputChannels, const float* input, float* output, 
      const float* filter1, const float* filter2, int filterSize
) {
   const int n2y = blockIdx.x;
   const int n2x = blockIdx.y;
   const int ch = threadIdx.x;

   if (ch > inputChannels) return;
   int toeplitzRowIdx = n2y*inputSize + n2x;
   int groupSizeX = inputSize+1;
   int groupSizeY = inputSize+1 - filterSize + 1;
   // int groupIdxY = toeplitzRowIdx / groupSizeY;
   // int groupIdxX = groupIdxY;
   int groupIdx = toeplitzRowIdx / groupSizeY;
   int zeroesLeft = toeplitzRowIdx % groupSizeY;
   // int zeroesRight = groupSizeX - filterSize - zeroesLeft;  

   float t = 0;
   for (int curGroup = groupIdx, fRow = 0; fRow < filterSize; curGroup++, fRow++) {
      
      for (int j = 0; j < filterSize; j++) {
         if (groupIdx == 0 && ch == 0) {
         printf("curGroup %d;  fRow %d;  j %d;\t f=%d;  i=%d\n", curGroup, fRow, j, filter1[idx2(ch, fRow, j, filterSize)], input[idx2(ch, curGroup, j, inputSize+1)]);
      }
         t += filter1[idx2(ch, fRow, j, filterSize)] * input[idx2(ch, curGroup,  j, inputSize+1)]; // ?
      }
   }
   // for (int i = groupIdx*groupSizeX + zeroesLeft, curGroup = groupIdx; i < inputSize; i += groupSizeX, curGroup++) {
   //    for (int j = 0; j < filterSize; j++) {
   //       int inpIdx = ch*inputSize*inputSize + curGroup*inputSize + j;
   //       t += filter1[idx2(ch, curGroup, j, filterSize)] * input[toeplitzRowIdx]; // ?
   //    }
   // }
   if (ch == 0 && groupIdx == 0) 
      printf("ch=%d; n2y=%d; n2x=%d; grIdx=%d; tRow=%d;grSize=(%d, %d)\t%d\n\n", ch, n2y, n2x, groupIdx, 
         toeplitzRowIdx, groupSizeX, groupSizeY, t);
   buffer[ch] = t;

   __syncthreads();

   if (ch >= outputChannels) return;
   t = 0;
   for (int ch2 = 0; ch2 < inputChannels; ch2++) {
      t += buffer[ch2] * filter2[f_idx(ch, ch2, inputChannels)];
   }
   output[idx(ch, n2x, n2y, outputChannels, inputSize, inputSize)] = ReLU(t);
}

bool check_error_status(cudaError_t status, const char *error_message) {
    if (status != cudaSuccess) {
        fprintf(stderr, error_message);
        return true;
    }
    return false;
}

bool test(int inputSize, int inputChannels, int outputChannels, int filterSize) {
   cudaError_t status;  
   // printf("test start\n");

   status = cudaSetDevice(0);
   if (check_error_status(status, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n"))
      return false;

      
   vector<float> hInput(inputChannels*inputSize*inputSize);
   vector<float> hOutput(outputChannels*inputSize*inputSize);
   vector<float> hFilter1(inputChannels*filterSize*filterSize);
   vector<float> hFilter2(inputChannels*outputChannels);

   for (int ch = 0; ch < inputChannels; ch++) {
      for (int y = 0; y < inputSize; y++) {
         for (int x = 0; x < inputSize; x++) {
            hInput[idx(ch, x, y, inputChannels, inputSize, inputSize)] = rand() % 3 + 1.0 / (1.0 + rand() % 3);
         }
      }
   }
   // printf("hInput +\n");
   for (int ch = 0; ch < inputChannels; ch++) {
      for (int y = 0; y < filterSize; y++) {
         for (int x = 0; x < filterSize; x++) {
            hFilter1[idx(ch, x, y, inputChannels, filterSize, filterSize)] = rand() % 3 + 1.0 / (1.0 + rand() % 3);
         }
      }
   }
   // printf("hFilter1 +\n");
   for (int oChannel = 0; oChannel < outputChannels; oChannel++) {
      for (int iChannel = 0; iChannel < inputChannels; iChannel++) {
         hFilter2[f_idx(iChannel, oChannel, inputChannels)] = rand() % 3 + 1.0 / (1.0 + rand() % 3);
      }
   }
   // printf("host data initialized\n");
   

   float *dInput, *dOutput, *dFilter1, *dFilter2;

   status = cudaMalloc((void**)&dInput, hInput.size()*sizeof(float));
   if (check_error_status(status, "cudaMalloc failed!\n"))
      return false;
   status = cudaMalloc((void**)&dOutput, hOutput.size()*sizeof(float));
   if (check_error_status(status, "cudaMalloc failed!\n"))
      return false;
   status = cudaMalloc((void**)&dFilter1, hFilter1.size()*sizeof(float));
   if (check_error_status(status, "cudaMalloc failed!\n"))
      return false;
   status = cudaMalloc((void**)&dFilter2, hFilter2.size()*sizeof(float));
   if (check_error_status(status, "cudaMalloc failed!\n"))
      return false;

   // printf("device data malloc success\n");

   status = cudaMemcpy(dInput, hInput.data(), hInput.size()*sizeof(float), cudaMemcpyHostToDevice);
   if (check_error_status(status, "cudaMemCpy failed!\n"))
      return false;
   status = cudaMemcpy(dFilter1, hFilter1.data(), hFilter1.size()*sizeof(float), cudaMemcpyHostToDevice);
   if (check_error_status(status, "cudaMemCpy failed!\n"))
      return false;
   status = cudaMemcpy(dFilter2, hFilter2.data(), hFilter2.size()*sizeof(float), cudaMemcpyHostToDevice);
   if (check_error_status(status, "cudaMemCpy failed!\n"))
      return false;

   // printf("device data memcpy success\n");

   // int blockSizeX = inputSize;
   // int blockSizeY = inputSize - filterSize + 1;
   // vector<float> hToeplitz(inputChannels*blockSizeX*blockSizeX*blockSizeY*blockSizeY);

   dim3 dimBlock(max(inputChannels, outputChannels), 1);
   dim3 dimGrid(inputSize, inputSize);
   // printf("foo started\n");
   depthwise_separable_convolution<<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(
      inputSize, inputChannels,
      outputChannels, dInput, dOutput, 
      dFilter1, dFilter2, filterSize);
   // printf("foo ended\n");

   status = cudaMemcpy(hOutput.data(), dOutput, hOutput.size()*sizeof(float), cudaMemcpyDeviceToHost);
   if (check_error_status(status, "couldn't load device output to host"))
      return false;
   // printf("output copy success\n");

   cudaFree(dInput);
   cudaFree(dOutput);
   cudaFree(dFilter1);
   cudaFree(dFilter2);

   return true;
}


void print_vector2d(vector<float> vec, int size, int channel) {
   printf("Channel %d\n", channel);
   for (int y = 0; y < size; y++) {
      for (int x = 0; x < size; x++) {
         printf("%f ", vec[idx2(channel, y, x, size)]);
      }
      printf("\n");
   }
}

bool static_test() {
   int inputChannels = 2, outputChannels = 2;
   int inputSize = 3, filterSize = 2;
   cudaError_t status;  

   status = cudaSetDevice(0);
   if (check_error_status(status, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n"))
      return false;
      
   vector<float> hInput(inputChannels*(inputSize+1)*(inputSize+1));
   vector<float> hOutput(outputChannels*inputSize*inputSize);
   vector<float> hFilter1(inputChannels*filterSize*filterSize);
   vector<float> hFilter2(inputChannels*outputChannels);

   for (int ch = 0; ch < inputChannels; ch++) {
      for (int y = 0; y < inputSize; y++) {
         for (int x = 0; x < inputSize; x++) {
            hInput[idx2(ch, y, x, inputSize+1)] = rand() % 3 + 1.0 / (1.0 + rand() % 3);
         }
      }
   }
   printf("hInput+\n");
   for (int ch = 0; ch < inputChannels; ch++) {
      for (int x = 0, y = inputSize; x < inputSize+1; x++) {
         hInput[idx2(ch, y, x, inputSize+1)] = 0;         
      }
      for (int x = inputSize, y = 0; y < inputSize+1; y++) {
         hInput[idx2(ch, y, x, inputSize+1)] = 0;         
      }
   }
   printf("padding+\n");
   printf("Input:\n");
   for (int ch = 0; ch < inputChannels; ch++) {
      print_vector2d(hInput, inputSize+1, ch);
   }
   // printf("hInput +\n");
   for (int ch = 0; ch < inputChannels; ch++) {
      for (int y = 0; y < filterSize; y++) {
         for (int x = 0; x < filterSize; x++) {
            hFilter1[idx(ch, x, y, inputChannels, filterSize, filterSize)] = rand() % 3 + 1.0 / (1.0 + rand() % 3);
         }
      }
   }
   printf("Filter 1:\n");
   for (int ch = 0; ch < inputChannels; ch++) {
      print_vector2d(hFilter1, filterSize, ch);
   }
   // printf("hFilter1 +\n");
   for (int oChannel = 0; oChannel < outputChannels; oChannel++) {
      for (int iChannel = 0; iChannel < inputChannels; iChannel++) {
         hFilter2[f_idx(iChannel, oChannel, inputChannels)] = rand() % 3 + 1.0 / (1.0 + rand() % 3);
      }
   }
   // printf("host data initialized\n");
   

   float *dInput, *dOutput, *dFilter1, *dFilter2;

   status = cudaMalloc((void**)&dInput, hInput.size()*sizeof(float));
   if (check_error_status(status, "cudaMalloc failed!\n"))
      return false;
   status = cudaMalloc((void**)&dOutput, hOutput.size()*sizeof(float));
   if (check_error_status(status, "cudaMalloc failed!\n"))
      return false;
   status = cudaMalloc((void**)&dFilter1, hFilter1.size()*sizeof(float));
   if (check_error_status(status, "cudaMalloc failed!\n"))
      return false;
   status = cudaMalloc((void**)&dFilter2, hFilter2.size()*sizeof(float));
   if (check_error_status(status, "cudaMalloc failed!\n"))
      return false;

   // printf("device data malloc success\n");

   status = cudaMemcpy(dInput, hInput.data(), hInput.size()*sizeof(float), cudaMemcpyHostToDevice);
   if (check_error_status(status, "cudaMemCpy failed!\n"))
      return false;
   status = cudaMemcpy(dFilter1, hFilter1.data(), hFilter1.size()*sizeof(float), cudaMemcpyHostToDevice);
   if (check_error_status(status, "cudaMemCpy failed!\n"))
      return false;
   status = cudaMemcpy(dFilter2, hFilter2.data(), hFilter2.size()*sizeof(float), cudaMemcpyHostToDevice);
   if (check_error_status(status, "cudaMemCpy failed!\n"))
      return false;

   // printf("device data memcpy success\n");

   // int blockSizeX = inputSize;
   // int blockSizeY = inputSize - filterSize + 1;
   // vector<float> hToeplitz(inputChannels*blockSizeX*blockSizeX*blockSizeY*blockSizeY);

   dim3 dimBlock(max(inputChannels, outputChannels), 1);
   dim3 dimGrid(inputSize, inputSize);
   // printf("foo started\n");
   depthwise_separable_convolution_toeplitz<<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(
      inputSize, inputChannels,
      outputChannels, dInput, dOutput, 
      dFilter1, dFilter2, filterSize);
   // printf("foo ended\n");

   status = cudaMemcpy(hOutput.data(), dOutput, hOutput.size()*sizeof(float), cudaMemcpyDeviceToHost);
   if (check_error_status(status, "couldn't load device output to host"))
      return false;

   printf("Output:\n");
   for (int ch = 0; ch < outputChannels; ch++) {
      print_vector2d(hOutput, inputSize, ch);
   }
   // printf("output copy success\n");

   cudaFree(dInput);
   cudaFree(dOutput);
   cudaFree(dFilter1);
   cudaFree(dFilter2);

   return true;
}

int main() {
   int inputSize = 1 << 10;
	int inputChannels = 3;
	int outputChannels = 16;
	int filterSize = 3;

   static_test();

   // int testNum = 1000;   
   // for (int i = 0; i < testNum; i++) {
   //    bool res = test(inputSize, inputChannels, outputChannels, filterSize);
   //    if (res == false) {
   //       printf("%d test fail\n", i);
   //       return -1;
   //    }
   //    printf("%d test success\n", i);
   // }   
}