#ifndef _ARG_PROC_H
#define _ARG_PROC_H

#define MAX_GPU_COUNT 100

#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

typedef struct ARGUTYPE
{

  char *big_sac;    // input list file of -A and -B
  char *stack_dir;  // output dir for CC vector
  int gpu_id;       // GPU ID
  int save_linear;  // 是否线性叠加结果
  int save_pws;     // 保留相位加权叠加结果
  int save_tfpws;   // 保留时频域相位加权叠加结果
  int gpu_task_num; // 同时开启的gpu数量
} ARGUTYPE;

void usage();
void ArgumentProcess(int argc, char **argv, ARGUTYPE *parg);
#endif