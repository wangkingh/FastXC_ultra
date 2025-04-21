#ifndef _MY_WRITE_H
#define _MY_WRITE_H
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "node_util.h"
#include "gen_ccfpath.h"
#include "sac.h"

// 模式定义
#define MODE_APPEND 1
#define MODE_AGGREGATE 2

int my_write_sac(const char *name, SACHEAD hd, const float *ar, int write_mode);

#endif