#ifndef PAR_WRITE_STEP_H
#define PAR_WRITE_STEP_H

#include "node_util.h"     /* PairBatchManager, PairNode, TimeData … */
#include "read_spec_lst.h" /* FilePaths           */
#include "par_write_sac.h" /* ThreadWritePool     */

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
                              ThreadWritePool *pool);

#endif
