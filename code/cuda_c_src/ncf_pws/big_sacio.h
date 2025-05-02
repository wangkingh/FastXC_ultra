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

/* ========== 扫描表条目（可扩展） ========== */
typedef struct
{
    off_t hdr_off;      /* header 绝对偏移           */
    off_t data_off;     /* data  绝对偏移            */
    unsigned char flag; /* 1=good, 0=bad          */
} OffItem;

int PreScanBigSac(const char *big_sac,
                  unsigned *pSegCount,  /* 段数            */
                  unsigned *pNpts,      /* 每段样点数      */
                  OffItem **pOffItems); /* malloc 数组     */

int ReadBigSac(const char *big_sac, float *pAllData,
               unsigned segment_count, unsigned data_count, SACHEAD *pFirstHd);
#endif