#ifndef _GEN_CCF_PATH_H
#define _GEN_CCF_PATH_H
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <libgen.h>
#include <errno.h>
#include <stdbool.h>
#include "config.h"
#include "sac.h"
#include "segspec.h"
#include "util.h"
#include "cal_dist.h"
#include "my_write_sac.h"

#define FIELD_LEN 16
#define YEAR_LEN 5
#define JDAY_LEN 4
#define HM_LEN 5

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

int mkdir_p(const char *path, mode_t mode);

int SplitFileName(const char *fname, const char *delimiter,
                  char *net, size_t net_sz,
                  char *sta, size_t sta_sz,
                  char *year, size_t year_sz,
                  char *jday, size_t jday_sz,
                  char *hm, size_t hm_sz,
                  char *chn, size_t chn_sz);

void SacheadProcess(SACHEAD *ncfhd,
                    float stla, float stlo, float evla, float evlo,
                    float Gcarc, float Az, float Baz, float Dist,
                    float delta, int ncc, float cclength, TimeData *time_info);

int GenCCFPath(char *ccf_path, size_t ccf_sz,
               const char *src_path, const char *sta_path,
               const char *output_dir, size_t queue_id);

#endif