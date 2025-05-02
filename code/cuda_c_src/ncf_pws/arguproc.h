#ifndef _ARG_PROC_H
#define _ARG_PROC_H

#define MAX_GPU_COUNT 100

#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

/*-------------------------------------------------------------
 * 命令行参数容器
 *-----------------------------------------------------------*/
typedef struct ARGUTYPE
{
  char *big_sac;   /* Big-SAC 文件路径            */
  char *stack_dir; /* 输出目录根                  */
  int gpu_id;      /* GPU 设备 ID                 */

  /* 保存选项 */
  int save_linear; /* 保存线性叠加结果            */
  int save_pws;    /* 保存 PWS 结果               */
  int save_tfpws;  /* 保存 TF-PWS 结果            */

  /* 性能控制 */
  int gpu_task_num; /* 并发 GPU 任务数             */

  /* 新增：子叠加参数 */
  int sub_stack_size;  /* 每几道先做一次叠加，<2 表示关闭 */
  char *src_info_file; /* 若为 NULL 则不过滤 */
} ARGUTYPE;

void usage(void);
void ArgumentProcess(int argc, char **argv, ARGUTYPE *parg);

#endif /* _ARG_PROC_H */