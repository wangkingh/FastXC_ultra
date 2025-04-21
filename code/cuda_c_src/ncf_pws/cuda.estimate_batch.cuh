#ifndef _CUDA_ESTEIMATE_BATCH_CUH
#define _CUDA_ESTEIMATE_BATCH_CUH
#include <assert.h>
#include <cstddef>
#include <cstdlib>
#include <cuComplex.h>
#include "cuda.util.cuh"
#include <stdio.h>

size_t EstimateGpuBatch_CC(size_t gpu_id, size_t fiexed_ram, size_t unitram,
                           int numType, int rank, int *n, int *inembed,
                           int istride, int idist, int *onembed, int ostride,
                           int odist, cufftType *typeArr);

size_t EstimateFreqBatchSize(
    int gpu_id,
    size_t num_trace,
    size_t npts_ncf,
    size_t nfreq,
    size_t gpu_task_num,
    float safety_factor);

#endif