#ifndef _READ_BIG_SAC_H
#define _READ_BIG_SAC_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include "sac.h"

// 声明
int PreReadBigSac(const char *big_sac, unsigned *pSegmentCount, unsigned *pDataCount);
int ReadBigSac(const char *big_sac, float *pAllData,
               unsigned segment_count, unsigned data_count, SACHEAD *pFirstHd);
#endif