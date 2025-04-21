#ifndef __ARG_PROC_H
#define __ARG_PROC_H

#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#define MAX_GPU_COUNT 100

typedef struct ARGUTYPE
{
  char *src_lst_path; /* -A */
  char *sta_lst_path; /* -B */
  char *ncf_dir;      /* -O */
  float cclength;     /* -C */

  /* 原先的单一 max_distance 替换为 distmin/distmax */
  float distmin;
  float distmax;

  /* 新增的方位角范围 */
  float azmin;
  float azmax;

  /* 新增：源信息文件 */
  char *srcinfo_file; /* -S */

  /* GPU/CPU 以及新的 queue_id */
  size_t gpu_id;       /* -G: GPU ID */
  size_t queue_id;     /* -Q: Queue ID，原先的 task_id 改名过来 */
  size_t gpu_task_num; /* -U: 在单个 GPU 上部署的任务数 */
  size_t cpu_count;    /* -T: 使用的 CPU 个数 */
  int write_mode;      /* -M: 写模式(0=traditional, 1=append, 2=aggregate) */

} ARGUTYPE;

void usage();
void ArgumentProcess(int argc, char **argv, ARGUTYPE *pargument);

#endif
