#ifndef PRESTACK_H
#define PRESTACK_H
#include <stdio.h>
#include <sys/types.h> /* off_t */
#include "big_sacio.h" /* OffItem */

/* ---------- OffGroup: 连续视图切片 ------- */
typedef struct
{
    const OffItem *arr; /* 指向 all[first] */
    unsigned size;      /* 本组元素数      */
} OffGroup;

/* ---------- API ---------------------------------------- */

/* A. 均匀切片（零拷贝）                                   */
int group_offsets_uniform(OffItem *all,
                          unsigned ntrace,
                          unsigned grp_sz,
                          OffGroup **pGroups,
                          unsigned *pNgroups);

/* B. 释放由 A 生成的 OffGroup 数组                         */
void free_groups(OffGroup *groups);

/* C. 叠加单组（假设全是好段）                              */
int stack_group_average(FILE *fp,
                        OffGroup *grp,
                        unsigned nsamples,
                        float *out_trace,
                        int do_byteswap);

/* D. 填充整个预叠加矩阵（矩阵由调用者预先 malloc）          */
int fill_prestack_matrix(FILE *fp,
                         OffGroup *groups,
                         unsigned ngroups,
                         unsigned nsamples,
                         int do_byteswap,
                         float *matrix_out);

#endif /* PRESTACK_H */
