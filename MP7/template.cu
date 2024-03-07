// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here

// kernel 1: float * to unsigned char *.
__global__ void cast(float *input, unsigned char *output, int width, int height) {
  
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  
  if (x < width && y < height) {
    int idx = blockIdx.z * width * height + (y * (width) + x);
    output[idx] = (unsigned char) (255 * input[idx]);
  }
  
}

// kernel 2: RGB image to GrayScale
__global__ void Rgb_Gray(unsigned char *input, unsigned char *output, int width, int height) {
  
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  
  if (x < width && y < height) {
    int idx = 3*(y * width + x);
    unsigned char r = input[idx];
		unsigned char g = input[idx + 1];
		unsigned char b = input[idx + 2];
    unsigned char grey = 0.21*r + 0.71*g + 0.07*b ;
		output[idx / 3] = grey;
  }
}

// kernel 3: computes the histogram 
__global__ void grayScaleToHistogram(unsigned char *input, unsigned int *output, int width, int height) {
        
  __shared__ unsigned int histogram[HISTOGRAM_LENGTH];
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int tIdx = threadIdx.x + threadIdx.y * blockDim.x;

  if (tIdx < HISTOGRAM_LENGTH) {
    histogram[tIdx] = 0;
  }
  __syncthreads();

  if (x < width && y < height) {
    int idx = y * (width) + x;
    //unsigned char val = input[idx];
    atomicAdd(&(histogram[input[idx]]), 1);
  }
  __syncthreads();

  if (tIdx < HISTOGRAM_LENGTH) {
    atomicAdd(&(output[tIdx]), histogram[tIdx]);
  }

}

// kernel 4: CDF of historgram 
__global__ void CDF(unsigned int *input, float *output, int width, int height) {

  __shared__ unsigned int cdf[HISTOGRAM_LENGTH];
  cdf[threadIdx.x] = input[threadIdx.x];

  for (unsigned int stride = 1; stride <= HISTOGRAM_LENGTH / 2; stride *= 2) {
    __syncthreads();
    int idx = (threadIdx.x + 1) * 2 * stride - 1;
    if (idx < HISTOGRAM_LENGTH) {
      cdf[idx] += cdf[idx - stride];
    }
  }

  for (int stride = HISTOGRAM_LENGTH / 4; stride > 0; stride /= 2) {
    __syncthreads();
    int idx = (threadIdx.x + 1) * 2 * stride - 1;
    if (idx + stride < HISTOGRAM_LENGTH) {
      cdf[idx + stride] += cdf[idx];
    }
  }

  __syncthreads();
  output[threadIdx.x] = cdf[threadIdx.x] / ((float) (width * height));
}

// kernel 5: correct color
__global__ void correctColor(unsigned char *input, float *cdf, int width, int height) {
  
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int idx = blockIdx.z * width * height + (y * (width) + x);
    unsigned char val = input[idx];

    float equalized = 255 * (cdf[val] - cdf[0]) / (1.0 - cdf[0]);
    unsigned char out = min(max(equalized, 0.0), 255.0);
    input[idx]   = out;
  }

}

// kernel 6: back to float
__global__ void castF(unsigned char *input, float *output, int width, int height) {
  
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  
  if (x < width && y < height) {
    int idx = blockIdx.z * width * height + (y * (width) + x);
    output[idx] = (float) (input[idx]/ 255.0);
  }
}


int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  unsigned char *deviceImageUC;
  unsigned char *deviceImageGrey;
  //unsigned char *devicecolor
  unsigned int *devicehist;
  float *devicecdf;
  float *devicefloat;


  //@@ Insert more code here

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here

  cudaMalloc((void**) &devicefloat,  imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void**) &deviceImageGrey, imageWidth * imageHeight *  sizeof(unsigned char));
  cudaMalloc((void**) &deviceImageUC, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
  cudaMalloc((void**) &devicehist, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void**) &devicecdf,HISTOGRAM_LENGTH * sizeof(float));

  cudaMemcpy(devicefloat, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
  //kernel 1
  dim3 dimGrid1(ceil(imageWidth/32.0), ceil(imageHeight/32.0), imageChannels);
  dim3 dimBlock1(32, 32, 1);

  cast<<<dimGrid1, dimBlock1>>>( devicefloat, deviceImageUC,imageWidth, imageHeight);
  cudaDeviceSynchronize();
  //kernel 2
  dim3 dimGrid2(ceil(imageWidth/32.0), ceil(imageHeight/32.0), 1);
  dim3 dimBlock2(32, 32, 1);

  Rgb_Gray<<<dimGrid2, dimBlock2>>>( deviceImageUC, deviceImageGrey,imageWidth, imageHeight);
  cudaDeviceSynchronize();

  //kernel 3
  dim3 dimGrid3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), 1);
  dim3 dimBlock3(32, 32, 1);

  grayScaleToHistogram<<<dimGrid3, dimBlock3>>>(deviceImageGrey, devicehist,imageWidth, imageHeight);
  cudaDeviceSynchronize();

  //kernel 4
  dim3 dimGrid4(1, 1, 1);
  dim3 dimBlock4(HISTOGRAM_LENGTH, 1, 1);
  CDF<<<dimGrid4, dimBlock4>>>(devicehist, devicecdf, imageWidth, imageHeight);
  //HistCDF<<<dimGrid4, dimBlock4>>>(devicecdf, devicehist, imageHeight*imageWidth);
  cudaDeviceSynchronize();

  //kernel 5
  dim3 dimGrid5(ceil(imageWidth/32.0), ceil(imageHeight/32.0), imageChannels);
  dim3 dimBlock5(32, 32, 1);
  correctColor<<<dimGrid5, dimBlock5>>>(deviceImageUC, devicecdf,imageWidth, imageHeight);
  cudaDeviceSynchronize();

  // kernel 6
  dim3 dimGrid6(ceil(imageWidth/32.0), ceil(imageHeight/32.0), imageChannels);
  dim3 dimBlock6(32, 32, 1);
  castF<<<dimGrid6, dimBlock6>>>(deviceImageUC, devicefloat,imageWidth, imageHeight);
  cudaDeviceSynchronize();

  //@@ insert code here

  cudaMemcpy(hostOutputImageData, devicefloat, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);

  wbSolution(args, outputImage);

  cudaFree (devicefloat);
  cudaFree (deviceImageGrey);
  cudaFree (deviceImageUC);
  cudaFree (devicehist);
  cudaFree (devicecdf);



  return 0;
}
