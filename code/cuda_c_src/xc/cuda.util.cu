#include "cuda.util.cuh"

const float RAMUPPERBOUND = 0.9;

void DimCompute1D(dim3 *pdimgrd, dim3 *pdimblk, size_t width)
{
  // 设置线程块大小，假设 BLOCKX1D 是预定义的每个线程块的线程数
  pdimblk->x = BLOCKX1D;

  // 计算所需的网格大小，确保能够覆盖所有的元素
  pdimgrd->x = (width + BLOCKX1D - 1) / BLOCKX1D;
}

void DimCompute(dim3 *pdimgrd, dim3 *pdimblk, size_t width, size_t height)
{
  pdimblk->x = BLOCKX;
  pdimblk->y = BLOCKY;

  pdimgrd->x = (width + BLOCKX - 1) / BLOCKX;
  pdimgrd->y = (height + BLOCKY - 1) / BLOCKY;
}

int EstimateGpuBatch(size_t gpu_id, int nspec, int nstep, int gpu_task_num)
{
  // print task_num
  printf("Task num: %d\n", gpu_task_num);
  size_t availram = QueryAvailGpuRam(gpu_id) / gpu_task_num;

  int nfft = 2 * (nspec - 1);

  size_t step = 100;     // 初始步长
  size_t min_step = 1;   // 最小步长
  size_t src_count = 0;  // 源/台 的数量
  size_t pair_count = 0; // 总共涉及多少台站对的计算
  size_t cufftram = 0;
  size_t req_input_ram = 0;
  size_t req_output_ram = 0;
  size_t req_final_output_ram = 0;
  size_t req_final_cc_ram = 0;
  size_t reqram = 0;

  int rank = 1;
  int n[1] = {nfft};
  int inembed[1] = {nfft};
  int onembed[1] = {nfft};
  int istride = 1;
  int idist = nfft;
  int ostride = 1;
  int odist = nfft;
  cufftType type = CUFFT_C2R;
  // print nstep and nspec, nfft, and pair_count
  printf("Nstep: %d\n", nstep);
  printf("Nspec: %d\n", nspec);
  printf("Nfft: %d\n", nfft);
  printf("Pair count: %ld\n", pair_count);
  printf("Cufft type: %d\n", type);
  while (true)
  {
    src_count += step;
    pair_count = src_count * src_count;
    req_input_ram = 2 * src_count * nspec * nstep * sizeof(cuComplex);
    req_output_ram = pair_count * nspec * nstep * sizeof(cuComplex);
    req_final_output_ram = pair_count * nspec * sizeof(cuComplex);
    req_final_cc_ram = pair_count * nfft * sizeof(float);
    cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride, odist, type, pair_count, &cufftram);
    reqram = req_input_ram + req_output_ram + req_final_output_ram + req_final_cc_ram + cufftram;
    // 打印各部分占用
    // printf("Src count: %ld\n", src_count);
    // printf("Input ram: %.3f GB\n", req_input_ram * 1.0 / (1L << 30));
    // printf("Output ram: %.3f GB\n", req_output_ram * 1.0 / (1L << 30));
    // printf("Final output ram: %.3f GB\n", req_final_output_ram * 1.0 / (1L << 30));
    // printf("Final cc ram: %.3f GB\n", req_final_cc_ram * 1.0 / (1L << 30));
    // printf("Cufft ram: %.3f GB\n", cufftram * 1.0 / (1L << 30));
    // printf("Total ram: %.3f GB\n", reqram * 1.0 / (1L << 30));
    // printf("-----------------------------\n");
    if (reqram > availram)
    {
      if (step > min_step)
      {
        src_count -= step; // 回退到安全配置
        step /= 2;         // 减小步长
        reqram = 0;        // 重置内存计算，避免使用错误的内存值
      }
      else
      {
        src_count -= step; // 最后尝试步长仍然过大，需要回退
        break;             // 退出循环
      }
    }
  }

  return src_count; // 返回成功
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

void GpuFree(void **pptr)
{
  CUDACHECK(cudaFree(*pptr));
  *pptr = NULL;
}