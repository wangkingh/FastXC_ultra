#include "cuda.xc_dual.cuh"
#include "cuda.util.cuh"
#include "segspec.h"
#include <cstddef>
#include <cstring>
#include <cuda_runtime.h>
#include <cufft.h>
#include <linux/limits.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#define min(x, y) ((x) < (y) ? (x) : (y))

extern "C"
{
#include "sac.h"
#include "arguproc.h"
#include "read_segspec.h"
#include "read_spec_lst.h"
#include "par_read_spec.h"
#include "par_write_sac.h"
#include "par_write_step.h"
#include "par_filter_nodes.h"
#include "gen_ccfpath.h"
#include "util.h"
}

int main(int argc, char **argv)
{
  // 1. 定义结构体，解析命令行参数
  ARGUTYPE cmd_args;
  ArgumentProcess(argc, argv, &cmd_args);

  // 2. 取出结构体内的各字段
  float cc_length = cmd_args.cclength;
  char *ncf_directory = cmd_args.ncf_dir;
  char *srcinfo_file = cmd_args.srcinfo_file;
  float dist_min = cmd_args.distmin;
  float dist_max = cmd_args.distmax;
  float az_min = cmd_args.azmin;
  float az_max = cmd_args.azmax;
  size_t gpu_id = cmd_args.gpu_id;
  size_t queue_id = cmd_args.queue_id; // 新增字段
  size_t gpu_task_count = cmd_args.gpu_task_num;
  size_t cpu_thread_count = cmd_args.cpu_count;
  int save_segment = cmd_args.save_segment;
  int write_mode = cmd_args.write_mode;

  // 设置当前要使用的 GPU
  CUDACHECK(cudaSetDevice(gpu_id));

  if (ncf_directory && *ncf_directory) /* 非空才尝试创建 */
  {
    if (mkdir_p(ncf_directory, 0755) != 0) /* 0755 = rwxr-xr-x */
    {
      /* 这里决定策略：直接返回错误、抛异常、或继续但打印警告 */
      fprintf(stderr,
              "ERROR: cannot create NCF output directory \"%s\": %s\n",
              ncf_directory, strerror(errno));
      return -1; /* 或者你的错误码 */
    }
  }

  // 读取源和台站的路径列表
  FilePaths *source_file_paths = read_spec_lst(cmd_args.src_lst_path);
  FilePaths *station_file_paths = read_spec_lst(cmd_args.sta_lst_path);

  // 读取首个源文件（只是为了获取一些头参数）
  SEGSPEC sample_spec_header;
  if (read_spechead(source_file_paths->paths[0], &sample_spec_header) != 0)
  {
    fprintf(stderr, "Error: Unable to read source file header.\n");
    return -1;
  }
  int num_frequency_points = sample_spec_header.nspec;
  int num_steps = sample_spec_header.nstep;
  float sampling_interval = sample_spec_header.dt;
  int fft_size = 2 * (num_frequency_points - 1);
  int half_cc_size = (int)floorf(cc_length / sampling_interval);
  int cc_size = 2 * half_cc_size + 1;

  // 估算主机和GPU的能力，以决定单次可以处理多少源/台
  size_t max_sta_num = EstimateGpuBatch(gpu_id, num_frequency_points, num_steps, gpu_task_count);
  printf("src count is %d, sta count is %d\n", source_file_paths->count, station_file_paths->count);
  max_sta_num = min(max_sta_num, (size_t)source_file_paths->count);
  max_sta_num = min(max_sta_num, (size_t)station_file_paths->count);
  printf("max_sta_num is %ld\n", max_sta_num);
  // max_sta_num /= 2;
  max_sta_num = max_sta_num > 0 ? max_sta_num : 1;
  size_t pairs_per_batch = max_sta_num * max_sta_num;
  printf("pair_batch is %ld\n", pairs_per_batch);

  // 计算分组数目，并生成“管理器”数组
  int source_group_count = (source_file_paths->count + max_sta_num - 1) / max_sta_num;
  int station_group_count = (station_file_paths->count + max_sta_num - 1) / max_sta_num;
  int num_pair_managers = source_group_count * station_group_count;

  // 3. 分配批次管理器数组
  PairBatchManager *batch_manager_array = NULL;
  CpuMalloc((void **)&batch_manager_array, num_pair_managers * sizeof(PairBatchManager));

  // 判断是否是单阵列
  size_t single_array_flag = strcmp(cmd_args.src_lst_path, cmd_args.sta_lst_path);
  int manager_idx = 0;
  int multi_array_flag = 0; // 标记是否存在双台阵

  for (int src_group_idx = 0; src_group_idx < source_group_count; src_group_idx++)
  {
    // 如果是单一阵列的情况，台站索引和源索引必须相同才处理
    int station_group_idx_start = 0;
    if (single_array_flag == 0)
    {
      station_group_idx_start = src_group_idx;
    }

    for (int sta_group_idx = station_group_idx_start; sta_group_idx < station_group_count; sta_group_idx++)
    {
      int src_start = src_group_idx * max_sta_num;
      int src_end = (src_start + max_sta_num > source_file_paths->count)
                        ? source_file_paths->count
                        : src_start + max_sta_num;

      int sta_start = sta_group_idx * max_sta_num;
      int sta_end = (sta_start + max_sta_num > station_file_paths->count)
                        ? station_file_paths->count
                        : sta_start + max_sta_num;

      // 4.1 当前 batch manager
      PairBatchManager *curr_manager = &batch_manager_array[manager_idx];
      curr_manager->src_start_idx = src_start;
      curr_manager->src_end_idx = src_end;
      curr_manager->sta_start_idx = sta_start;
      curr_manager->sta_end_idx = sta_end;

      // 判断是否单台阵
      if ((single_array_flag == 0) && (src_start == sta_start) && (src_end == sta_end))
      {
        curr_manager->is_single_array = 1;
      }
      else
      {
        curr_manager->is_single_array = 0;
        multi_array_flag = 1;
      }

      // 4.2 统计本批节点数量 (index_count)
      size_t index_count = 0;
      for (size_t s = src_start; s < (size_t)src_end; s++)
      {
        size_t sta_start_tmp = curr_manager->is_single_array ? s : sta_start;
        for (size_t t = sta_start_tmp; t < (size_t)sta_end; t++)
        {
          index_count++;
        }
      }

      curr_manager->node_count = index_count;

      // 4.3 分配节点数组
      PairNode *node_array = NULL;
      CpuMalloc((void **)&node_array, index_count * sizeof(PairNode));
      curr_manager->pair_node_array = node_array;

      // 4.4 填充节点数组中的相对索引
      size_t node_pos = 0;
      for (size_t s = src_start; s < (size_t)src_end; s++)
      {
        size_t sta_start_tmp = curr_manager->is_single_array ? s : sta_start;
        for (size_t t = sta_start_tmp; t < (size_t)sta_end; t++)
        {
          PairNode *pNode = &node_array[node_pos++];
          pNode->source_relative_idx = s - src_start;
          pNode->station_relative_idx = t - sta_start;

          // 其余字段先初始化
          pNode->station_lat = 0.0f;
          pNode->station_lon = 0.0f;
          pNode->source_lat = 0.0f;
          pNode->source_lon = 0.0f;
          pNode->great_circle_dist = 0.0f;
          pNode->azimuth = 0.0f;
          pNode->back_azimuth = 0.0f;
          pNode->linear_distance = 0.0f;

          pNode->time_info.year = 0;
          pNode->time_info.day_of_year = 0;
          pNode->time_info.hour = 0;
          pNode->time_info.minute = 0;

          pNode->valid_flag = 0;         // 默认 0，等后续筛选
          pNode->step_valid_flag = NULL; // 若后面需要分配，可在 Filter 步骤分配
        }
      }

      manager_idx++;
    }
  }
  num_pair_managers = manager_idx;

  // 1. 创建并初始化筛选条件 FilterCriteria
  FilterCriteria criteria;
  criteria.dist_min = dist_min; // 来自命令行
  criteria.dist_max = dist_max;
  criteria.az_min = az_min;
  criteria.az_max = az_max;

  // 2. 创建线程池
  ThreadPoolFilter *filter_pool = create_threadpool_filter_nodes(cpu_thread_count);

  // 3. 并行过滤 + 压缩
  for (int idx = 0; idx < num_pair_managers; idx++)
  {
    FilterNodeParallel(&batch_manager_array[idx],
                       source_file_paths,
                       station_file_paths,
                       filter_pool,
                       &criteria);

    CompressBatchManager(&batch_manager_array[idx]);
  }

  destroy_threadpool_filter_nodes(filter_pool);

  // 为数据分配空间
  size_t vec_count = num_steps * num_frequency_points;
  size_t vec_byte_size = vec_count * sizeof(complex);

  complex *h_source_buffer = NULL;
  cuComplex *d_source_buffer = NULL;
  CpuMalloc((void **)&h_source_buffer, max_sta_num * vec_byte_size);
  GpuMalloc((void **)&d_source_buffer, max_sta_num * vec_byte_size);

  complex *h_station_buffer = NULL;
  cuComplex *d_station_buffer = NULL;
  if (multi_array_flag == 1) // 如果存在双台阵，需要分配 station
  {
    CpuMalloc((void **)&h_station_buffer, max_sta_num * vec_byte_size);
    GpuMalloc((void **)&d_station_buffer, max_sta_num * vec_byte_size);
  }

  float *h_crosscor_time = NULL; // CPU端互相关时域缓冲
  CpuMalloc((void **)&h_crosscor_time, pairs_per_batch * cc_size * sizeof(float));

  // 创建 FFT 相关的配置
  dim3 dimGrid_1D, dimBlock_1D;
  dim3 dimGrid_2D, dimBlock_2D;
  cufftHandle plan;
  int rank = 1;
  int n[1] = {fft_size};
  int inembed = fft_size / 2 + 1;
  int istride = 1, idist = fft_size / 2 + 1;
  int onembed = fft_size;
  int ostride = 1, odist = fft_size;
  cufftType type = CUFFT_C2R;

  CufftPlanAlloc(&plan, rank, n, &inembed, istride, idist,
                 &onembed, ostride, odist, type, pairs_per_batch);

  // GPU上的下标列表、互相关缓冲等
  size_t *d_src_idx_list = NULL;
  size_t *d_sta_idx_list = NULL;
  cuComplex *d_crosscor_buffer = NULL;
  cuComplex *d_crosscor_stack = NULL;
  int *d_sign_vector = NULL;
  float *d_crosscor_time = NULL;

  GpuMalloc((void **)&d_src_idx_list, pairs_per_batch * sizeof(size_t));
  GpuMalloc((void **)&d_sta_idx_list, pairs_per_batch * sizeof(size_t));
  GpuMalloc((void **)&d_crosscor_buffer, pairs_per_batch * vec_byte_size);
  GpuMalloc((void **)&d_crosscor_stack, pairs_per_batch * num_frequency_points * sizeof(cuComplex));
  GpuMalloc((void **)&d_crosscor_time, pairs_per_batch * fft_size * sizeof(float));

  // 计算相移向量
  GpuCalloc((void **)&d_sign_vector, num_frequency_points * sizeof(int));
  DimCompute1D(&dimGrid_1D, &dimBlock_1D, num_frequency_points);
  generateSignVector<<<dimGrid_1D, dimBlock_1D>>>(d_sign_vector, num_frequency_points);

  // 线程池（读文件、写文件）
  ThreadPoolRead *read_pool = create_threadpool_read(cpu_thread_count);
  ThreadWritePool *pool_plain = NULL;
  ThreadWriteStepPool *pool_step = NULL;
  if (save_segment == 0)
  {
    pool_plain = create_threadwrite_pool(cpu_thread_count);
  }
  else
  {
    pool_step = create_write_step_pool(cpu_thread_count * gpu_task_count);
  }

  // 用于判断是否需要重新读源/台数据
  size_t src_start_flag = 0;
  size_t src_end_flag = 0;
  size_t sta_start_flag = 0;
  size_t sta_end_flag = 0;

  // 查看可用显存
  size_t available_gpu_ram = QueryAvailGpuRam(gpu_id);
  available_gpu_ram = available_gpu_ram / 1024 / 1024 / 1024;
  printf("availram is %ld\n", available_gpu_ram);

  // 逐个 manager 处理
  for (size_t m_idx = 0; m_idx < (size_t)num_pair_managers; m_idx++)
  {
    PairBatchManager *curr_batch_mgr = &batch_manager_array[m_idx];
    size_t node_count = curr_batch_mgr->node_count;
    if (node_count == 0)
    {
      continue;
    }
    size_t *src_relative_idx_host = (size_t *)malloc(node_count * sizeof(size_t));
    size_t *sta_relative_idx_host = (size_t *)malloc(node_count * sizeof(size_t));
    PairNode *node_array = curr_batch_mgr->pair_node_array;
    for (size_t i = 0; i < node_count; i++)
    {
      src_relative_idx_host[i] = node_array[i].source_relative_idx;
      sta_relative_idx_host[i] = node_array[i].station_relative_idx;
    }
    // 拷到 GPU
    CUDACHECK(cudaMemcpy(d_src_idx_list,
                         src_relative_idx_host,
                         node_count * sizeof(size_t),
                         cudaMemcpyHostToDevice));

    CUDACHECK(cudaMemcpy(d_sta_idx_list,
                         sta_relative_idx_host,
                         node_count * sizeof(size_t),
                         cudaMemcpyHostToDevice));

    free(src_relative_idx_host);
    free(sta_relative_idx_host);

    size_t src_start = curr_batch_mgr->src_start_idx;
    size_t src_end = curr_batch_mgr->src_end_idx;
    size_t sta_start = curr_batch_mgr->sta_start_idx;
    size_t sta_end = curr_batch_mgr->sta_end_idx;

    size_t src_trace_num = src_end - src_start;
    size_t sta_trace_num = sta_end - sta_start;

    // 清空输出缓存
    memset(h_crosscor_time, 0, pairs_per_batch * cc_size * sizeof(float));
    CUDACHECK(cudaMemset(d_crosscor_stack, 0, pairs_per_batch * num_frequency_points * sizeof(cuComplex)));
    CUDACHECK(cudaMemset(d_crosscor_time, 0, pairs_per_batch * fft_size * sizeof(float)));

    DimCompute(&dimGrid_2D, &dimBlock_2D, vec_count, node_count);

    // 若源数据有变化，重新读取
    if (src_start != src_start_flag || src_end != src_end_flag)
    {
      ReadSpecArrayParallel(source_file_paths,
                            h_source_buffer,
                            src_start, src_end,
                            vec_count, read_pool);

      CUDACHECK(cudaMemcpy(d_source_buffer,
                           h_source_buffer,
                           src_trace_num * vec_byte_size,
                           cudaMemcpyHostToDevice));

      src_start_flag = src_start;
      src_end_flag = src_end;
    }

    if (curr_batch_mgr->is_single_array == 0)
    {
      // 如果是双台阵，需要读台数据
      if (sta_start != sta_start_flag || sta_end != sta_end_flag)
      {
        ReadSpecArrayParallel(station_file_paths,
                              h_station_buffer,
                              sta_start, sta_end,
                              vec_count, read_pool);

        CUDACHECK(cudaMemcpy(d_station_buffer,
                             h_station_buffer,
                             sta_trace_num * vec_byte_size,
                             cudaMemcpyHostToDevice));

        sta_start_flag = sta_start;
        sta_end_flag = sta_end;
      }

      cmultiply2DKernel<<<dimGrid_2D, dimBlock_2D>>>(d_source_buffer, d_src_idx_list,
                                                     d_station_buffer, d_sta_idx_list,
                                                     d_crosscor_buffer,
                                                     node_count, vec_count);
    }
    else
    {
      // 单台阵 => 源与台相同
      cmultiply2DKernel<<<dimGrid_2D, dimBlock_2D>>>(d_source_buffer, d_src_idx_list,
                                                     d_source_buffer, d_sta_idx_list,
                                                     d_crosscor_buffer,
                                                     node_count, vec_count);
    }

    if (save_segment == 0)
    {

      printf("node count is %ld, pair batch is %ld\n", node_count, pairs_per_batch);
      DimCompute(&dimGrid_2D, &dimBlock_2D, num_frequency_points, node_count);
      for (size_t step_idx = 0; step_idx < (size_t)num_steps; step_idx++)
      {
        csum2DKernel<<<dimGrid_2D, dimBlock_2D>>>(d_crosscor_stack,
                                                  num_frequency_points,
                                                  d_crosscor_buffer,
                                                  num_frequency_points,
                                                  num_frequency_points,
                                                  node_count,
                                                  step_idx,
                                                  num_steps);
      }

      // 相移
      DimCompute(&dimGrid_2D, &dimBlock_2D, num_frequency_points, node_count);
      applyPhaseShiftKernel<<<dimGrid_2D, dimBlock_2D>>>(d_crosscor_stack,
                                                         d_sign_vector,
                                                         num_frequency_points,
                                                         num_frequency_points,
                                                         node_count);

      // IFFT
      cufftExecC2R(plan, (cufftComplex *)d_crosscor_stack, (cufftReal *)d_crosscor_time);

      // 归一化
      DimCompute(&dimGrid_2D, &dimBlock_2D, fft_size, node_count);
      InvNormalize2DKernel<<<dimGrid_2D, dimBlock_2D>>>(d_crosscor_time,
                                                        fft_size,
                                                        fft_size,
                                                        node_count,
                                                        sampling_interval);

      // 拷回并写出，只截取时序中间的 cc_size
      CUDACHECK(cudaMemcpy2D(h_crosscor_time,
                             cc_size * sizeof(float),
                             d_crosscor_time + (num_frequency_points - half_cc_size - 1),
                             fft_size * sizeof(float),
                             cc_size * sizeof(float),
                             node_count,
                             cudaMemcpyDeviceToHost));

      // 写出互相关结果
      write_pairs_parallel(curr_batch_mgr,
                           source_file_paths,
                           station_file_paths,
                           h_crosscor_time,
                           sampling_interval,
                           cc_size,
                           cc_length,
                           ncf_directory,
                           queue_id,
                           write_mode,
                           pool_plain);
    }
    else
    {
      /* ► 计算 launch 配置一次即可 */
      DimCompute(&dimGrid_2D, &dimBlock_2D, num_frequency_points, node_count);

      for (size_t step_idx = 0; step_idx < (size_t)num_steps; ++step_idx)
      {
        CUDACHECK(cudaMemcpy2D(d_crosscor_stack, num_frequency_points * sizeof(cuComplex),
                               d_crosscor_buffer + step_idx * num_frequency_points, num_frequency_points * sizeof(cuComplex) * num_steps,
                               num_frequency_points, node_count,
                               cudaMemcpyDeviceToDevice));
        /* 2. 相移 */
        applyPhaseShiftKernel<<<dimGrid_2D, dimBlock_2D>>>(
            d_crosscor_stack,
            d_sign_vector,
            num_frequency_points,
            num_frequency_points,
            node_count);

        /* 3. IFFT : d_crosscor_stack(C2R) -> d_crosscor_time */
        cufftExecC2R(plan,
                     (cufftComplex *)d_crosscor_stack,
                     (cufftReal *)d_crosscor_time);

        /* 4. 归一化 */
        DimCompute(&dimGrid_2D, &dimBlock_2D, fft_size, node_count);
        InvNormalize2DKernel<<<dimGrid_2D, dimBlock_2D>>>(
            d_crosscor_time,
            fft_size, fft_size,
            node_count,
            sampling_interval);

        /* 5. 只取中心 cc_size 栅格拷回 host */
        CUDACHECK(cudaMemcpy2D(h_crosscor_time,
                               cc_size * sizeof(float),
                               d_crosscor_time + (num_frequency_points - half_cc_size - 1),
                               fft_size * sizeof(float),
                               cc_size * sizeof(float),
                               node_count,
                               cudaMemcpyDeviceToHost));

        float step_len = num_frequency_points * sampling_interval;
        write_pairs_step_parallel(curr_batch_mgr,
                                  source_file_paths, station_file_paths,
                                  h_crosscor_time,
                                  sampling_interval,
                                  cc_size,
                                  cc_length,
                                  step_len, /* ← new */
                                  ncf_directory,
                                  queue_id,
                                  write_mode,
                                  step_idx, /* 当前 step */
                                  pool_step);
      } /* for step */
    }
  }

  // 销毁线程池
  destroy_threadpool_read(read_pool);
  destroy_threadwrite_pool(pool_plain);
  destroy_write_step_pool(pool_step);

  for (int idx = 0; idx < num_pair_managers; idx++)
  {
    // 每个批次分配过 pair_node_array，就在这里逐一释放
    free(batch_manager_array[idx].pair_node_array);
  }
  // 最后释放 batch_manager_array 本身
  free(batch_manager_array);

  // 释放其余 CPU 资源
  CpuFree((void **)&h_crosscor_time);
  CpuFree((void **)&h_source_buffer);
  CpuFree((void **)&h_station_buffer);
  freeFilePaths(source_file_paths);
  freeFilePaths(station_file_paths);

  // 释放 GPU 端内存
  GpuFree((void **)&d_source_buffer);
  GpuFree((void **)&d_station_buffer);
  GpuFree((void **)&d_src_idx_list);
  GpuFree((void **)&d_sta_idx_list);
  GpuFree((void **)&d_crosscor_buffer);
  GpuFree((void **)&d_crosscor_stack);
  GpuFree((void **)&d_sign_vector);
  GpuFree((void **)&d_crosscor_time);

  // 销毁 CUFFT plan
  CUFFTCHECK(cufftDestroy(plan));

  return 0;
}