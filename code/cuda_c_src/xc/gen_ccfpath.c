#include "gen_ccfpath.h"

int mkdir_p(const char *path, mode_t mode)
{
    if (!path || *path == '\0')
    {
        errno = EINVAL;
        return -1;
    }

    /* ---------- 1. 复制路径到可写缓冲 ---------- */
    char tmp[PATH_MAX];
    size_t len = strnlen(path, PATH_MAX);
    if (len >= PATH_MAX)
    {
        errno = ENAMETOOLONG;
        return -1;
    }

    memcpy(tmp, path, len + 1); /* 连同 '\0' */

    /* ---------- 2. 逐级 mkdir ---------- */
    for (char *p = tmp + 1; *p; ++p)
    {
        if (*p == '/')
        {
            *p = '\0';
            if (mkdir(tmp, mode) == -1 && errno != EEXIST)
                return -1; /* 真失败 */
            *p = '/';
        }
    }
    /* ---------- 3. mkdir 最终目录 ---------- */
    if (mkdir(tmp, mode) == -1 && errno != EEXIST)
        return -1;

    return 0;
}

/* Split the file name — portable, no strlcpy dependency */
int SplitFileName(const char *fname, const char *delimiter,
                  char *net, size_t net_sz,
                  char *sta, size_t sta_sz,
                  char *year, size_t year_sz,
                  char *jday, size_t jday_sz,
                  char *hm, size_t hm_sz,
                  char *chn, size_t chn_sz)
{
    if (!fname || !delimiter || !net || !sta || !year ||
        !jday || !hm || !chn)
        return -1; /* 参数空 */

    char *copy = my_strdup(fname);
    if (!copy)
        return -2; /* 内存不足 */

    char *save = NULL;
    const size_t cap[6] = {net_sz, sta_sz, year_sz,
                           jday_sz, hm_sz, chn_sz};
    char *out[6] = {net, sta, year, jday, hm, chn};

    for (int i = 0; i < 6; ++i)
    {
        const char *tok = strtok_r(i == 0 ? copy : NULL, delimiter, &save);
        if (!tok)
        { /* 缺字段 */
            free(copy);
            return -(10 + i);
        }
        /* snprintf：若截断，目标串已 NUL 终结；上层可以按需要再检查 */
        snprintf(out[i], cap[i], "%s", tok);
    }

    free(copy);
    return 0;
}

void SacheadProcess(SACHEAD *ncfhd,
                    float stla, float stlo, float evla, float evlo,
                    float Gcarc, float Az, float Baz, float Dist,
                    float delta, int ncc, float cclength, TimeData *time_info)
{
    *ncfhd = sac_null;
    /* Write in stla,stlo,evla,evlo*/
    ncfhd->stla = stla;
    ncfhd->stlo = stlo;
    ncfhd->evla = evla;
    ncfhd->evlo = evlo;

    // Convert back to float after the function call
    ncfhd->gcarc = Gcarc;
    ncfhd->az = Az;
    ncfhd->baz = Baz;
    ncfhd->dist = Dist;

    /* necessary header info */
    ncfhd->iftype = 1;
    ncfhd->leven = 1;
    ncfhd->delta = delta;
    /* npts of hd should b ncfnpts+1, eg 2k+1 */
    ncfhd->npts = ncc;
    ncfhd->b = -1.0 * cclength;
    ncfhd->e = cclength;
    ncfhd->unused27 = 1;
    /* set o time to be zero */
    ncfhd->o = 0.0;

    // 5) Set the time info
    if (time_info)
    {
        ncfhd->nzyear = time_info->year;
        ncfhd->nzjday = time_info->day_of_year;
        ncfhd->nzhour = time_info->hour;
        ncfhd->nzmin = time_info->minute;
        ncfhd->nzsec = 0;
        ncfhd->nzmsec = 0;
    }

    /* END OF MAKE COMMON HEADER INFO */
}

/* 把 path 的文件名拷到 dst，保证 NUL 结尾；超长即 ENAMETOOLONG */
static int copy_basename(char *dst, size_t dst_sz, const char *path)
{
    if (!dst || !path)
        return EINVAL;

    char tmp[PATH_MAX];
    if (snprintf(tmp, sizeof(tmp), "%s", path) >= (int)sizeof(tmp))
        return ENAMETOOLONG;

    const char *base = basename(tmp); /* libc 可能返回 "."，但此处已知是文件 */
    if (snprintf(dst, dst_sz, "%s", base) >= (int)dst_sz)
        return ENAMETOOLONG;

    return 0;
}

int GenCCFPath(char *ccf_path, size_t ccf_sz,
               const char *src_path, const char *sta_path,
               const char *output_dir, size_t queue_id)
{
    if (!ccf_path || !src_path || !sta_path || !output_dir)
        return -EINVAL;

    /* ---------- 1. 提取文件名 ---------- */
    char src_file[MAXNAME], sta_file[MAXNAME];
    int rc;

    rc = copy_basename(src_file, sizeof(src_file), src_path);
    if (rc)
        return -rc;
    rc = copy_basename(sta_file, sizeof(sta_file), sta_path);
    if (rc)
        return -rc;

    /* ---------- 2. 拆字段 ---------- */
    char src_net[FIELD_LEN], src_sta[FIELD_LEN], src_chn[FIELD_LEN];
    char sta_net[FIELD_LEN], sta_sta[FIELD_LEN], sta_chn[FIELD_LEN];
    char src_year[YEAR_LEN], src_jd[JDAY_LEN], src_hm[HM_LEN];
    char sta_year[YEAR_LEN], sta_jd[JDAY_LEN], sta_hm[HM_LEN];

    if (SplitFileName(src_file, ".", /* 安全版 SplitFileName */
                      src_net, sizeof(src_net),
                      src_sta, sizeof(src_sta),
                      src_year, sizeof(src_year),
                      src_jd, sizeof(src_jd),
                      src_hm, sizeof(src_hm),
                      src_chn, sizeof(src_chn)) != 0)
        return -EINVAL; /* 源文件名格式错误 */

    if (SplitFileName(sta_file, ".",
                      sta_net, sizeof(sta_net),
                      sta_sta, sizeof(sta_sta),
                      sta_year, sizeof(sta_year),
                      sta_jd, sizeof(sta_jd),
                      sta_hm, sizeof(sta_hm),
                      sta_chn, sizeof(sta_chn)) != 0)
        return -EINVAL; /* 目标文件名格式错误 */

    /* ---------- 3. 生成目录 ---------- */
    char ccf_dir[PATH_MAX];
    if (snprintf(ccf_dir, sizeof(ccf_dir), "%s/queue_%zu", output_dir, queue_id) >= (int)sizeof(ccf_dir))
        return -ENAMETOOLONG; /* 路径被截断 */

    if (mkdir_p(ccf_dir, 0755) != 0) /* 改用无 TOCTOU 版本 */
        return -errno;               /* 让上层决定打印还是重试 */

    /* ---------- 4. 生成文件名 ---------- */
    char ccf_name[MAXNAME];
    if (snprintf(ccf_name, sizeof(ccf_name), "%s-%s.%s-%s.%s-%s.bigsac",
                 src_net, sta_net, src_sta, sta_sta, src_chn, sta_chn) >= (int)sizeof(ccf_name))
        return -ENAMETOOLONG;

    /* ---------- 5. 拼完整路径 ---------- */
    if (snprintf(ccf_path, ccf_sz, "%s/%s", ccf_dir, ccf_name) >= (int)ccf_sz)
        return -ENAMETOOLONG;

    return 0;
}