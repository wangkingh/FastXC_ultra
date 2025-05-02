#ifndef __ARG_PROC_H
#define __ARG_PROC_H

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#define MAX_GPU_COUNT 100

typedef struct ARGUTYPE
{
  /* 文件与参数 */
  char *src_lst_path; /* -A */
  char *sta_lst_path; /* -B */
  char *ncf_dir;      /* -O */
  float cclength;     /* -C seconds */

  float distmin, distmax; /* -D km */
  float azmin, azmax;     /* -Z deg */

  char *srcinfo_file; /* -S <file>  —— **恢复** */

  /* 运行环境 */
  size_t gpu_id;       /* -G */
  size_t queue_id;     /* -Q */
  size_t gpu_task_num; /* -U */
  size_t cpu_count;    /* -T */

  int write_mode;   /* -M 0/1/2 */
  int save_segment; /* -R : 1=输出分段，0=默认叠加+IFFT */
} ARGUTYPE;

void usage(void);
void ArgumentProcess(int argc, char **argv, ARGUTYPE *parg);

#endif
