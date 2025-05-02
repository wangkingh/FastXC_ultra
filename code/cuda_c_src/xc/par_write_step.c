/* ---------------------------------------------------------------------------
 * par_write_step.c  ——  按 “step-by-step” 写出互相关时序
 *
 * 功能：把单段（step）互相关序列写成 SAC 文件，文件名额外带 .stepXXXX
 *       并在 SAC 头 user0 字段里记录 “单段持续时间 step_len (sec)”，
 *       方便后续还原绝对时间轴。
 * --------------------------------------------------------------------------*/

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h> /* MAXPATH */

#include "node_util.h"     /* PairBatchManager / PairNode / TimeData */
#include "par_write_sac.h" /* SacheadProcess / my_write_sac */
#include "gen_ccfpath.h"   /* GenCCFPath            */
#include "read_spec_lst.h" /* FilePaths             */

/* 复用现有写线程池结构，但给 tinfo 扩展两个字段 */
typedef struct thread_info_write_step
{
    /* —— 老字段 —— */
    PairBatchManager *batch_mgr;
    FilePaths *src_path_list;
    FilePaths *sta_path_list;
    size_t start; /* [start, end) 节点区间 */
    size_t end;
    float *ncf_buffer; /* node_count × ncc      */
    float delta;
    int ncc;
    float cc_half_len;
    char *output_dir;
    int write_mode;
    size_t queue_id;

    /* —— 新增字段 —— */
    float step_len;  /* 单段持续时间 (sec) */
    size_t step_idx; /* 当前 step 序号    */
} thread_info_write_step;

/* 写线程主体 -------------------------------------------------------------- */
static void *write_file_step(void *arg)
{
    thread_info_write_step *info = (thread_info_write_step *)arg;
    PairBatchManager *mgr = info->batch_mgr;
    PairNode *node_array = mgr->pair_node_array;

    for (size_t k = info->start; k < info->end; ++k)
    {
        PairNode *node = &node_array[k];

        /* 绝对索引 → 找到源/台文件路径 */
        size_t gsrc = mgr->src_start_idx + node->source_relative_idx;
        size_t gsta = mgr->sta_start_idx + node->station_relative_idx;
        char *src_path = info->src_path_list->paths[gsrc];
        char *sta_path = info->sta_path_list->paths[gsta];

        /* 生成基准文件名（目录 /SRC.STA.sac 之类） */
        char ncf_path[MAXPATH];
        if (GenCCFPath(ncf_path, sizeof(ncf_path),
                       src_path, sta_path,
                       info->output_dir,
                       info->queue_id) != 0)
        {
            fprintf(stderr, "GenCCFPath failed for %s & %s\n", src_path, sta_path);
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

        int seg_sec = (int)round(info->step_len); /* 3600 */

        double off_d = (double)info->step_idx * seg_sec; /* 精确整数秒 */
        hd.user3 = (float)info->step_len;                /* 仍存原值 */
        hd.user4 = (float)off_d;                         /* 绝不再飘 */

        /* 写数据 */
        float *trace = info->ncf_buffer + k * info->ncc;
        if (my_write_sac(ncf_path, hd, trace, info->write_mode) != 0)
        {
            fprintf(stderr, "Write SAC failed: %s\n", ncf_path);
        }
    }
    return NULL;
}

/* 并行调度 --------------------------------------------------------------- */
int write_pairs_step_parallel(PairBatchManager *batch_mgr,
                              FilePaths *src_path_list,
                              FilePaths *sta_path_list,
                              float *cc_buf,
                              float delta,
                              int ncc,
                              float cc_half_len,
                              float step_len, /* ★ 单段长度 */
                              char *output_dir,
                              size_t queue_id,
                              int write_mode,
                              size_t step_idx, /* ★ 当前 step */
                              ThreadWritePool *pool)
{
    size_t total_nodes = batch_mgr->node_count;
    if (total_nodes == 0)
        return 0;

    size_t num_th = pool->num_threads;
    size_t per_th = total_nodes / num_th;
    size_t rem = total_nodes % num_th;

    size_t begin = 0;
    for (size_t i = 0; i < num_th; ++i)
    {
        thread_info_write_step *ti = (thread_info_write_step *)&pool->tinfo[i];

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

        /* 新字段 */
        ti->step_len = step_len;
        ti->step_idx = step_idx;

        begin = ti->end;

        if (pthread_create(&pool->threads[i],
                           NULL,
                           write_file_step,
                           ti))
        {
            fprintf(stderr, "Error creating write thread\n");
            return -1;
        }
    }

    /* join */
    for (size_t i = 0; i < num_th; ++i)
    {
        if (pthread_join(pool->threads[i], NULL))
        {
            fprintf(stderr, "Error joining write thread\n");
            return -1;
        }
    }
    return 0;
}
