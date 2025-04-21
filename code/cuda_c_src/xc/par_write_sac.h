#ifndef _PAR_WRITE_H
#define _PAR_WRITE_H
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "node_util.h"
#include "gen_ccfpath.h"
#include "my_write_sac.h"

typedef struct
{
    PairBatchManager *batch_mgr; ///< 指向本批次 (替换原先的 PAIRLIST_MANAGER)
    FilePaths *src_path_list;    ///< 源文件路径列表
    FilePaths *sta_path_list;    ///< 台文件路径列表
    size_t start;                ///< 处理的起始节点索引 (在 pair_node_array)
    size_t end;                  ///< 处理的结束节点索引
    float *ncf_buffer;           ///< 存储交叉相关函数结果的缓冲区 (时域序列)
    float delta;                 ///< 互相关采样间隔
    int ncc;                     ///< 交叉相关数据点数量
    float cc_length;             ///< 半个互相关长度
    char *output_dir;            ///< 输出文件路径
    int write_mode;              ///< 写入模式 (0=append, 1=aggregate, etc.)
    size_t queue_id;             ///< 任务 ID (用于生成文件名等)
} thread_info_write;

typedef struct
{
    pthread_t *threads;
    thread_info_write *tinfo;
    size_t num_threads;
} ThreadWritePool;

#ifdef __cplusplus
extern "C"
{
#endif

    ThreadWritePool *create_threadwrite_pool(size_t num_threads);

    void destroy_threadwrite_pool(ThreadWritePool *pool);

    int my_write_sac(const char *name,
                     SACHEAD hd,
                     const float *ar, int mode);

    /**
     * 并行写出函数
     *
     * @param batch_mgr   当前批次的管理器 (内含节点数组)
     * @param src_path_list 源文件路径列表
     * @param sta_path_list 台文件路径列表
     * @param ncf_buffer   完整的互相关结果缓冲(时域)
     * @param delta        采样间隔
     * @param ncc          互相关序列长度
     * @param cc_length    半个互相关长度
     * @param output_dir   输出目录
     * @param gpu_id       GPU编号
     * @param task_id      任务编号
     * @param write_mode   写入模式
     * @param pool         写线程池
     */
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
                             ThreadWritePool *pool);

#ifdef __cplusplus
}
#endif
#endif
