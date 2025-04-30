#ifndef _GEN_NCF_PATH_H
#define _GEN_NCF_PATH_H
#define MAXLINE 8192
#define MAXPATH 8192
#define MAXNAME 1024
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include "sac.h"

void CreateDir(char *sPathName);

void SplitFileName(const char *fname,
                   const char *delimiter,
                   char *net_pair_str,
                   char *sta_pair_str,
                   char *cmp_pair_str,
                   char *suffix);

#endif