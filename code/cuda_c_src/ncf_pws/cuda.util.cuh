#ifndef _CUDA_UTIL_CUH
#define _CUDA_UTIL_CUH

#include <assert.h>
#include <cstddef>
#include <cstdlib>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// 3060 gpu do not have enough resources, so I try modify the block size to 16
#define BLOCKX1D 256
#define BLOCKX2D 16
#define BLOCKY2D 16
#define BLOCKX3D 8
#define BLOCKY3D 8
#define BLOCKZ3D 8
#define BLOCKMAX 1024
#define BLOCKX 16
#define BLOCKY 16

#define CUDACHECK(cmd)                                              \
  do                                                                \
  {                                                                 \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess)                                           \
    {                                                               \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

#define CUFFTCHECK(cmd)                                                \
  do                                                                   \
  {                                                                    \
    cufftResult_t e = cmd;                                             \
    if (e != CUFFT_SUCCESS)                                            \
    {                                                                  \
      printf("Failed: CuFFT error %s:%d %d\n", __FILE__, __LINE__, e); \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

size_t QueryAvailGpuRam(size_t gpu_id);

void DimCompute(dim3 *, dim3 *, size_t, size_t);

/* Compute the block and dimension for cuda kernel fucntion*/
void DimCompute1D(dim3 *pdimgrd, dim3 *pdimblk, size_t width);
void DimCompute2D(dim3 *, dim3 *, size_t, size_t);
void DimCompute3D(dim3 *pdimgrd, dim3 *pdimblk, size_t width, size_t height, size_t depth);

void CufftPlanAlloc(cufftHandle *, int, int *, int *, int, int, int *, int, int,
                    cufftType, int);
void GpuMalloc(void **, size_t);
void GpuCalloc(void **, size_t);
void GpuFree(void **);

#endif