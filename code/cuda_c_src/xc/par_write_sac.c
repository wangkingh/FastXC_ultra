#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "node_util.h"
#include "par_write_sac.h"

ThreadWritePool *create_threadwrite_pool(size_t num_threads)
{
    ThreadWritePool *pool = malloc(sizeof(ThreadWritePool));
    if (!pool)
    {
        fprintf(stderr, "Memory allocation failed for thread pool\n");
        return NULL;
    }
    pool->threads = malloc(num_threads * sizeof(pthread_t));
    pool->tinfo = malloc(num_threads * sizeof(thread_info_write));
    if (!pool->threads || !pool->tinfo)
    {
        free(pool->threads);
        free(pool->tinfo);
        free(pool);
        fprintf(stderr, "Memory allocation failed for threads or thread info\n");
        return NULL;
    }
    pool->num_threads = num_threads;
    return pool;
}

void destroy_threadwrite_pool(ThreadWritePool *pool)
{
    if (pool)
    {
        free(pool->threads);
        free(pool->tinfo);
        free(pool);
    }
}

// 实际写文件的线程函数
static void *write_file(void *arg)
{
    thread_info_write *write_info = (thread_info_write *)arg;

    // 1) 取出批次管理器和节点数组
    PairBatchManager *batch_mgr = write_info->batch_mgr;
    PairNode *node_array = batch_mgr->pair_node_array;
    size_t src_start_idx = batch_mgr->src_start_idx;
    size_t sta_start_idx = batch_mgr->sta_start_idx;

    // 2) 取出写相关参数
    FilePaths *src_path_list = write_info->src_path_list;
    FilePaths *sta_path_list = write_info->sta_path_list;
    float *ncf_buf = write_info->ncf_buffer;
    float delta = write_info->delta;
    int ncc = write_info->ncc;
    float cc_length = write_info->cc_length;
    char *output_dir = write_info->output_dir;
    int write_mode = write_info->write_mode;
    size_t queue_id = write_info->queue_id;

    // 3) 节点区间
    size_t start_node_idx = write_info->start;
    size_t end_node_idx = write_info->end;

    // 4) 遍历节点
    for (size_t node_idx = start_node_idx; node_idx < end_node_idx; node_idx++)
    {
        // 4.1 拿到节点
        PairNode *node = &node_array[node_idx];

        // 计算该节点在 ncf_buf 里的偏移
        size_t buffer_offset = node_idx * ncc;

        // 4.2 绝对索引: 用于定位全局文件路径
        size_t global_src_idx = src_start_idx + node->source_relative_idx;
        size_t global_sta_idx = sta_start_idx + node->station_relative_idx;

        // 4.3 获取源、台路径
        char *src_path = src_path_list->paths[global_src_idx];
        char *sta_path = sta_path_list->paths[global_sta_idx];

        // 4.4 准备 SAC 头信息
        SACHEAD ncf_hd;
        // lat/lon/distance from node
        float stla = node->station_lat;
        float stlo = node->station_lon;
        float evla = node->source_lat;
        float evlo = node->source_lon;
        float gcarc = node->great_circle_dist;
        float az = node->azimuth;
        float baz = node->back_azimuth;
        float dist = node->linear_distance;
        TimeData tinfo = node->time_info; // 如果你要写时间也在这里

        // 4.5 生成输出文件路径
        char ncf_path[MAXPATH];
        int ret = GenCCFPath(ncf_path,           /* 输出缓冲区           */
                             sizeof(ncf_path),   /* 缓冲区大小 (size_t) */
                             src_path, sta_path, /* 源/目标 SAC 路径    */
                             output_dir,         /* 输出根目录          */
                             queue_id);          /* 队列号              */

        if (ret != 0)
        {
            fprintf(stderr, "GenCCFPath failed: %s\n", strerror(-ret));
            /* 按需要退出或处理 */
        }

        // 4.6 构造 SAC header (你可用 SacheadProcess 或自己写)
        SacheadProcess(&ncf_hd, stla, stlo, evla, evlo,
                       gcarc, az, baz, dist,
                       delta, ncc, cc_length,
                       &tinfo);

        // 4.7 写 SAC
        // ncf_buf + buffer_offset 就是该节点的互相关序列
        printf("Writing %s\n", ncf_path);
        my_write_sac(ncf_path, ncf_hd, ncf_buf + buffer_offset, write_mode);
    }

    return NULL;
}

// 并行写函数
int write_pairs_parallel(PairBatchManager *batch_mgr,
                         FilePaths *src_path_list,
                         FilePaths *sta_path_list,
                         float *ncf_buffer,
                         float delta,
                         int ncc,
                         float cc_length,
                         char *output_dir,
                         size_t queue_id,
                         int write_mode,
                         ThreadWritePool *pool)
{
    // 1) 获取节点总数
    size_t total_nodes = batch_mgr->node_count;
    if (total_nodes == 0)
        return 0; // 没节点就不用写

    // 2) 平均分配到线程
    size_t nodes_per_thread = total_nodes / pool->num_threads;
    size_t remainder = total_nodes % pool->num_threads;

    size_t start_idx = 0;
    for (int i = 0; i < (int)pool->num_threads; i++)
    {
        pool->tinfo[i].batch_mgr = batch_mgr;
        pool->tinfo[i].src_path_list = src_path_list;
        pool->tinfo[i].sta_path_list = sta_path_list;
        pool->tinfo[i].start = start_idx;
        pool->tinfo[i].end = start_idx + nodes_per_thread + (i < remainder ? 1 : 0);
        pool->tinfo[i].ncf_buffer = ncf_buffer;
        pool->tinfo[i].delta = delta;
        pool->tinfo[i].ncc = ncc;
        pool->tinfo[i].cc_length = cc_length;
        pool->tinfo[i].output_dir = output_dir;
        pool->tinfo[i].write_mode = write_mode;
        pool->tinfo[i].queue_id = queue_id;

        start_idx = pool->tinfo[i].end;

        // 创建写线程
        if (pthread_create(&pool->threads[i], NULL, write_file, &pool->tinfo[i]))
        {
            fprintf(stderr, "Error creating write thread\n");
            return -1;
        }
    }

    // join
    for (int i = 0; i < (int)pool->num_threads; i++)
    {
        if (pthread_join(pool->threads[i], NULL))
        {
            fprintf(stderr, "Error joining write thread\n");
            return -1;
        }
    }

    return 0;
}