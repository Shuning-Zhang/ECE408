// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len) {
  // @@ Modify the body of this function to complete the functionality of
  // @@ the scan on the device
  // @@ You may need multiple kernel calls; write your kernels before this
  // @@ function and call them from the host
  __shared__ float B[BLOCK_SIZE * 2];

    int stride;
    int index;

    unsigned int t = threadIdx.x;
    unsigned int start = 2*blockIdx.x*blockDim.x;

    // check boundraries
    if (start + 2 * t < len) {
      B[2*t] = input[start+2*t];
    } else {
      B[2*t] = 0;
    }
    if (start + 2 * t + 1< len) {
      B[2 * t+ 1] = input[start+ 2 * t+ 1];
    } else {
      B[2 * t+ 1] = 0;
    }

    // Reduection
    for (stride = 1; stride <= blockDim.x; stride *= 2) {
      __syncthreads();
      int index = (threadIdx.x+1) * 2* stride -1;
      if (index < (BLOCK_SIZE * 2)) {
        B[index] += B[index - stride];
      }
    }
    
    stride = BLOCK_SIZE / 2;
    while (stride > 0)
    {
        __syncthreads();
        index = (t + 1) * stride * 2 - 1;
        if ((index + stride) < (BLOCK_SIZE * 2))
            B[index + stride] += B[index];
        stride = stride / 2;
    }

    __syncthreads();


    // copy back
    if (start + 2 * t < len) {
      input[start+ 2*t] = B[2*t];
    } 
    if (start + 2 * t + 1< len) {
      input[start+ 2 * t+ 1] = B[2 * t+ 1];
    } 

    // if scan is only partially done
    if ((len > (BLOCK_SIZE * 2)) && (t == 0))
        output[blockIdx.x] = B[(BLOCK_SIZE * 2) - 1];

}


__global__ void scanAdd(float *input, float *output, int len)
{
    unsigned int start = blockIdx.x * blockDim.x;
    unsigned int idx = start + threadIdx.x;

    if (idx < len && blockIdx.x >0){
      output[idx] += input[blockIdx.x - 1];
    }
}


int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *devicehelperBuffer;
  int numElements; // number of elements in the list


  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&devicehelperBuffer, ceil(numElements/(2.0 * BLOCK_SIZE))* sizeof(float)));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  wbTime_start(Compute, "Performing CUDA computation");

  dim3 DimGrid(ceil(numElements/(BLOCK_SIZE *2.0)), 1, 1);
  dim3 DimBlock(BLOCK_SIZE, 1, 1);
  scan<<<DimGrid, DimBlock>>>(deviceInput, devicehelperBuffer, numElements);

  dim3 HelpArrayDimGrid(1, 1, 1);
  scan<<<HelpArrayDimGrid, DimBlock>>>(devicehelperBuffer, NULL, ceil(numElements/(BLOCK_SIZE *2.0)));

  dim3 AddDimBlock(BLOCK_SIZE << 1, 1, 1);
  scanAdd<<<DimGrid, AddDimBlock>>>(devicehelperBuffer, deviceInput, numElements);  

  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceInput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(devicehelperBuffer);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

