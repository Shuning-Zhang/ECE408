#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)


//@@ Define any useful program-wide constants here
#define MASK_WIDTH  3
#define TILE_WIDTH  8
//@@ Define constant memory for device kernel here
__constant__ float m_dk[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  const int size_s = TILE_WIDTH + MASK_WIDTH -1;
  __shared__ float S_M[size_s][size_s][size_s];

  int tx = threadIdx.x; 
  int ty = threadIdx.y; 
  int tz = threadIdx.z;

  int x_o = blockIdx.x * TILE_WIDTH + tx;
  int y_o = blockIdx.y * TILE_WIDTH + ty;
  int z_o = blockIdx.z * TILE_WIDTH + tz;

  int x_i = x_o - (MASK_WIDTH/2);
  int y_i = y_o - (MASK_WIDTH/2);
  int z_i = z_o - (MASK_WIDTH/2);

  //check bounds
  if (
    (x_i < x_size) && (x_i >= 0) &&
    (y_i < y_size) && (y_i >= 0) &&
    (z_i < z_size) && (z_i >= 0)
  ){
    S_M[tz][ty][tx] = input[(z_i * (y_size * x_size)) + (y_i * x_size) + x_i];
  }
  //out off bound
  else{
    S_M[tz][ty][tx] = 0.0;
  }

  __syncthreads();

  if (tx <TILE_WIDTH && ty < TILE_WIDTH && tz < TILE_WIDTH && x_o < x_size && y_o < y_size && z_o < z_size){
    float Pvalue = 0;
    for(int i = 0; i < MASK_WIDTH; i++){
      for(int j = 0; j < MASK_WIDTH; j++){
        for(int k = 0; k < MASK_WIDTH; k++){
            Pvalue += m_dk[i][j][k] * S_M[tz + i][ty + j][tx + k];
        }
      }
    }
    output[z_o * (y_size * x_size) + y_o * (x_size) + x_o] = Pvalue;
  }

}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions


  cudaMalloc((void **)&deviceInput, (inputLength-3) * sizeof(float));
  cudaMalloc((void **)&deviceOutput, (inputLength-3) * sizeof(float));


  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu

  cudaMemcpy(deviceInput, &hostInput[3], (inputLength-3) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(m_dk, hostKernel, kernelLength * sizeof(float));

  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here

  dim3 DimGrid(ceil((1.0*x_size)/TILE_WIDTH), ceil((1.0*y_size)/TILE_WIDTH), ceil((1.0*z_size)/TILE_WIDTH));
  dim3 DimBlock((TILE_WIDTH + MASK_WIDTH - 1),(TILE_WIDTH + MASK_WIDTH - 1),(TILE_WIDTH + MASK_WIDTH - 1));

  //@@ Launch the GPU kernel here
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);

  wbTime_start(Copy, "Copying data from the GPU");

  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbTime_stop(Copy, "Copying data from the GPU");

  cudaMemcpy(&hostOutput[3], deviceOutput, (inputLength-3) * sizeof(float), cudaMemcpyDeviceToHost);

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
