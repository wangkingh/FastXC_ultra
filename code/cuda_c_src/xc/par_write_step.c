#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "par_write_step.h"
#include "gen_ccfpath.h"

ThreadWriteStepPool *create_write_step_pool(size_t num_threads)
{
    ThreadWriteStepPool *pool = malloc(sizeof(ThreadWriteStepPool));
    if (!pool)
    {
        perror("malloc pool");
        return NULL;
    }

    pool->threads = calloc(num_threads, sizeof(pthread_t));
    pool->tinfo = calloc(num_threads, sizeof(thread_info_write_step));
    if (!pool->threads || !pool->tinfo)
    {
        perror("calloc threads/tinfo");
        free(pool->threads);
        free(pool->tinfo);
        free(pool);
        return NULL;
    }

    pool->num_threads = num_threads;
    return pool;
}

void destroy_write_step_pool(ThreadWriteStepPool *pool)
{
    if (!pool)
        return;
    free(pool->threads);
    free(pool->tinfo);
    free(pool);
}

/* ---------- 写线程主体 ---------- */
static void *write_file_step(void *arg)
{
    thread_info_write_step *info = arg;
    PairBatchManager *mgr = info->batch_mgr;
    PairNode *node_array = mgr->pair_node_array;

    for (size_t k = info->start; k < info->end; ++k)
    {
        PairNode *node = &node_array[k];

        /* 绝对索引 → 路径 */
        size_t gsrc = mgr->src_start_idx + node->source_relative_idx;
        size_t gsta = mgr->sta_start_idx + node->station_relative_idx;
        char *src_path = info->src_path_list->paths[gsrc];
        char *sta_path = info->sta_path_list->paths[gsta];

        /* 生成输出文件名 */
        char ncf_path[PATH_MAX];
        if (GenCCFPath(ncf_path, sizeof(ncf_path),
                       src_path, sta_path,
                       info->output_dir,
                       info->queue_id) != 0)
        {
            fprintf(stderr, "GenCCFPath failed for %s %s\n", src_path, sta_path);
            continue;
        }

        /* 组装 SAC 头 */
        SACHEAD hd;
        SacheadProcess(&hd,
                       node->station_lat, node->station_lon,
                       node->source_lat, node->source_lon,
                       node->great_circle_dist,
                       node->azimuth, node->back_azimuth,
                       node->linear_distance,
                       info->delta,
                       info->ncc,
                       info->cc_half_len,
                       &node->time_info);

        int seg_sec = (int)round(info->step_len);
        double off_d = (double)info->step_idx * seg_sec;
        hd.user3 = info->step_len;
        hd.user4 = off_d;

        /* 写数据 */
        float *trace = info->ncf_buffer + k * info->ncc;
        if (my_write_sac(ncf_path, hd, trace, info->write_mode) != 0)
            fprintf(stderr, "Write SAC failed: %s\n", ncf_path);
    }
    return NULL;
}

/* ---------- 并行调度 ---------- */
int write_pairs_step_parallel(PairBatchManager *batch_mgr,
                              FilePaths *src_path_list,
                              FilePaths *sta_path_list,
                              float *cc_buf,
                              float delta,
                              int ncc,
                              float cc_half_len,
                              float step_len,
                              char *output_dir,
                              size_t queue_id,
                              int write_mode,
                              size_t step_idx,
                              ThreadWriteStepPool *pool)
{
    size_t total_nodes = batch_mgr->node_count;
    if (total_nodes == 0)
        return 0;

    size_t num_th = pool->num_threads;
    if (num_th == 0)
    {
        fprintf(stderr, "ThreadWriteStepPool.num_threads == 0\n");
        return -1;
    }

    /* 均分任务 */
    size_t per_th = total_nodes / num_th;
    size_t rem = total_nodes % num_th;

    size_t begin = 0;
    for (size_t i = 0; i < num_th; ++i)
    {
        thread_info_write_step *ti = &pool->tinfo[i];

        ti->batch_mgr = batch_mgr;
        ti->src_path_list = src_path_list;
        ti->sta_path_list = sta_path_list;
        ti->start = begin;
        ti->end = begin + per_th + (i < rem ? 1 : 0);
        ti->ncf_buffer = cc_buf;
        ti->delta = delta;
        ti->ncc = ncc;
        ti->cc_half_len = cc_half_len;
        ti->output_dir = output_dir;
        ti->write_mode = write_mode;
        ti->queue_id = queue_id;
        ti->step_len = step_len;
        ti->step_idx = step_idx;

        begin = ti->end;

        if (pthread_create(&pool->threads[i], NULL,
                           write_file_step, ti))
        {
            perror("pthread_create");
            return -1;
        }
    }

    for (size_t i = 0; i < num_th; ++i)
        if (pthread_join(pool->threads[i], NULL))
        {
            perror("pthread_join");
            return -1;
        }

    return 0;
}
