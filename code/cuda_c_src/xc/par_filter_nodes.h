#ifndef _PAR_FILTER_NODELIST
#define _PAR_FILTER_NODELIST

#include <stdlib.h>
#include <pthread.h>
#include <limits.h>
#include <stdio.h>
#include "node_util.h"
#include "read_segspec.h"
#include "segspec.h"
#include "util.h"
#include "cal_dist.h"

typedef struct
{
    float dist_min; ///< 距离下限
    float dist_max; ///< 距离上限
    float az_min;   ///< 方位角下限
    float az_max;   ///< 方位角上限

    // 如果“源信息”是更多字段，也可加入这里
    // float source_lat;
    // float source_lon;
    // ...
} FilterCriteria;

/**
 * @struct thread_info_filter
 * @brief 每个线程处理一段节点区间、访问一个 PairBatchManager，
 *        并根据 FilterCriteria 做筛选。
 */
typedef struct
{
    FilePaths *srcFileList;      // 源文件列表
    FilePaths *staFileList;      // 台文件列表
    PairBatchManager *batch_mgr; // 指向某个批次的管理器
    size_t start;                // 节点起始索引 (在 pair_node_array 里)
    size_t end;                  // 节点结束索引 (在 pair_node_array 里)
    FilterCriteria *criteria;    // 筛选准则 (距离上下限、方位角上下限等)
} thread_info_filter;

// 线程池结构体定义
typedef struct
{
    pthread_t *threads;
    thread_info_filter *tinfo;
    size_t num_threads;
} ThreadPoolFilter;

#ifdef __cplusplus
extern "C"
{
#endif
    ThreadPoolFilter *create_threadpool_filter_nodes(size_t num_threads);

    void destroy_threadpool_filter_nodes(ThreadPoolFilter *pool);

    // 并行筛选
    int FilterNodeParallel(PairBatchManager *batch_mgr,
                           FilePaths *src_paths, FilePaths *sta_paths,
                           ThreadPoolFilter *pool,
                           FilterCriteria *criteria);

    // 压缩：将 valid_flag=1 的节点挤到前面
    void CompressBatchManager(PairBatchManager *batch_mgr);

#ifdef __cplusplus
}
#endif
#endif