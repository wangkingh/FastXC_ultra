#ifndef PAR_WRITE_STEP_H
#define PAR_WRITE_STEP_H

#include "node_util.h"     /* PairBatchManager, PairNode, TimeData … */
#include "read_spec_lst.h" /* FilePaths           */

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

typedef struct
{
    pthread_t *threads;
    thread_info_write_step *tinfo;
    size_t num_threads;
} ThreadWriteStepPool;

/* ───────── 线程池管理 ───────── */
/* 创建写线程池（成功返回非 NULL；失败返回 NULL） */
ThreadWriteStepPool *create_write_step_pool(size_t num_threads);

/* 释放线程池（安全接受 NULL 指针） */
void destroy_write_step_pool(ThreadWriteStepPool *pool);

/* 新函数声明 */
int write_pairs_step_parallel(PairBatchManager *batch_mgr,
                              FilePaths *src_path_list,
                              FilePaths *sta_path_list,
                              float *cc_buf,     /* node_count × cc_size */
                              float delta,       /* 采样间隔 */
                              int ncc,           /* cc_size              */
                              float cc_half_len, /* 半长度，秒           */
                              float step_len,    /* ★ 单段时长，秒      */
                              char *output_dir,
                              size_t queue_id,
                              int write_mode,
                              size_t step_idx, /* 当前 step 序号      */
                              ThreadWriteStepPool *pool);

#endif
