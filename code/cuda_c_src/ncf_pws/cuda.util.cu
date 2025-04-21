#include "cuda.util.cuh"
#include <unistd.h>

const float RAMUPPERBOUND = 0.9;

// DimCompute: BLOCKX = 32, BLOCKY = 32
void DimCompute(dim3 *pdimgrd, dim3 *pdimblk, size_t width, size_t height)
{
  pdimblk->x = BLOCKX;
  pdimblk->y = BLOCKY;

  // for debug, trying to limit dimgrd
  pdimgrd->x = (width + BLOCKX - 1) / BLOCKX;
  pdimgrd->y = (height + BLOCKY - 1) / BLOCKY;
}

size_t QueryAvailGpuRam(size_t deviceID)
{
  size_t freeram, totalram;
  cudaSetDevice(deviceID);
  CUDACHECK(cudaMemGetInfo(&freeram, &totalram));
  freeram *= RAMUPPERBOUND;

  const size_t gigabytes = 1L << 30;
  printf("Avail gpu ram: %.3f GB\n", freeram * 1.0 / gigabytes);
  return freeram;
}

void DimCompute1D(dim3 *pdimgrd, dim3 *pdimblk, size_t width)
{
  // 设置线程块大小，假设 BLOCKX1D 是预定义的每个线程块的线程数
  pdimblk->x = BLOCKX1D;

  // 计算所需的网格大小，确保能够覆盖所有的元素
  pdimgrd->x = (width + BLOCKX1D - 1) / BLOCKX1D;
}

void DimCompute2D(dim3 *pdimgrd, dim3 *pdimblk, size_t width, size_t height)
{
  pdimblk->x = BLOCKX2D;
  pdimblk->y = BLOCKY2D;

  pdimgrd->x = (width + BLOCKX2D - 1) / BLOCKX2D;
  pdimgrd->y = (height + BLOCKY2D - 1) / BLOCKY2D;
}

void DimCompute3D(dim3 *pdimgrd, dim3 *pdimblk, size_t width, size_t height, size_t depth)
{
  // 设置每个维度的块大小
  pdimblk->x = BLOCKX3D;
  pdimblk->y = BLOCKY3D;
  pdimblk->z = BLOCKZ3D;

  // 计算每个维度所需的网格大小
  pdimgrd->x = (width + BLOCKX3D - 1) / BLOCKX3D;  // 计算X维度的网格数
  pdimgrd->y = (height + BLOCKY3D - 1) / BLOCKY3D; // 计算Y维度的网格数
  pdimgrd->z = (depth + BLOCKZ3D - 1) / BLOCKZ3D;  // 计算Z维度的网格数
}

void GpuFree(void **pptr)
{
  if (*pptr != NULL)
  {
    cudaFree(*pptr);
    *pptr = NULL;
  }
}

void CufftPlanAlloc(cufftHandle *pHandle, int rank, int *n, int *inembed,
                    int istride, int idist, int *onembed, int ostride,
                    int odist, cufftType type, int batch)
{
  // create cufft plan
  CUFFTCHECK(cufftPlanMany(pHandle, rank, n, inembed, istride, idist, onembed,
                           ostride, odist, type, batch));
}

void GpuMalloc(void **pptr, size_t sz) { CUDACHECK(cudaMalloc(pptr, sz)); }

void GpuCalloc(void **pptr, size_t sz)
{
  CUDACHECK(cudaMalloc(pptr, sz));

  CUDACHECK(cudaMemset(*pptr, 0, sz));
}