#include "big_sacio.h"

static unsigned get_data_count(const SACHEAD *hd)
{
    unsigned data_count = hd->npts;
    if (hd->iftype == IXY)
    {
        data_count = hd->npts * 2; // SAC 的复数/矢量类型
    }
    return data_count;
}

/*---------------------------------------------------------------
 * 扫描 Big-SAC，只记录每段数据区偏移量，不读进波形.
 *--------------------------------------------------------------*/
int PreScanBigSac(const char *big_sac,
                  unsigned *pSegCount,
                  unsigned *pDataCount,
                  OffItem **pOffItems)
{
    FILE *fp = fopen(big_sac, "rb");
    if (!fp)
    {
        fprintf(stderr, "[PreScanBigSac] open %s: %s\n",
                big_sac, strerror(errno));
        return -1;
    }

    /* ---- dynamic OffItem array ---- */
    unsigned cap = 128;
    OffItem *items = malloc(cap * sizeof *items);
    if (!items)
    {
        perror("malloc");
        fclose(fp);
        return -1;
    }

    /* ---- read first header ---- */
    off_t hdr_off = 0; /* ftello == 0 */
    SACHEAD hd_first;
    if (fread(&hd_first, sizeof hd_first, 1, fp) != 1)
    {
        fprintf(stderr, "[PreScanBigSac] read first header failed\n");
        free(items);
        fclose(fp);
        return -1;
    }
#ifdef BYTE_SWAP
    swab4((char *)&hd_first, sizeof hd_first);
#endif

    unsigned base_npts = get_data_count(&hd_first);
    if (base_npts == 0)
    {
        fprintf(stderr, "[PreScanBigSac] npts=0 in first segment\n");
        free(items);
        fclose(fp);
        return -1;
    }

    /* 保存首段 OffItem */
    items[0].hdr_off = hdr_off;
    items[0].data_off = hdr_off + sizeof(SACHEAD);
    items[0].flag = 1;

    const off_t data_bytes = (off_t)base_npts * sizeof(float);

    /* skip first data */
    if (fseeko(fp, data_bytes, SEEK_CUR) != 0)
    {
        perror("[PreScanBigSac] fseeko");
        free(items);
        fclose(fp);
        return -1;
    }

    /* ---- scan remaining segments ---- */
    unsigned seg_cnt = 1;
    while (1)
    {
        hdr_off = ftello(fp); /* next header pos (64-bit) */
        SACHEAD hd_tmp;

        if (fread(&hd_tmp, sizeof hd_tmp, 1, fp) != 1)
        {
            if (feof(fp))
                break; /* normal EOF */
            perror("[PreScanBigSac] fread");
            free(items);
            fclose(fp);
            return -1;
        }
#ifdef BYTE_SWAP
        swab4((char *)&hd_tmp, sizeof hd_tmp);
#endif
        if (get_data_count(&hd_tmp) != base_npts)
        {
            fprintf(stderr,
                    "[PreScanBigSac] seg#%u npts mismatch (%u vs %u)\n",
                    seg_cnt + 1, base_npts, get_data_count(&hd_tmp));
            free(items);
            fclose(fp);
            return -1;
        }

        /* grow OffItem array when needed */
        if (seg_cnt == cap)
        {
            cap <<= 1;
            OffItem *tmp = realloc(items, cap * sizeof *tmp);
            if (!tmp)
            {
                perror("realloc");
                free(items);
                fclose(fp);
                return -1;
            }
            items = tmp;
        }

        /* fullfill OffItem */
        items[seg_cnt].hdr_off = hdr_off;
        items[seg_cnt].data_off = hdr_off + sizeof(SACHEAD);
        items[seg_cnt].flag = 1; /* 先全部标为可用 */

        /* skip data block */
        if (fseeko(fp, data_bytes, SEEK_CUR) != 0)
        {
            perror("[PreScanBigSac] fseeko");
            free(items);
            fclose(fp);
            return -1;
        }
        ++seg_cnt;
    }
    fclose(fp);

    /* ---- shrink to exact size ---- */
    OffItem *final = realloc(items, seg_cnt * sizeof *items);
    if (!final)
        final = items; /* rare, keep original pointer */

    /* ---- outputs ---- */
    *pSegCount = seg_cnt;
    *pDataCount = base_npts;
    *pOffItems = final;
    return 0;
}

/**
 * @brief ReadBigSac: 真正读取 big_sac 中的多段数据到 pAllData
 *
 * @param big_sac        [输入] big_sac 文件
 * @param pAllData       [输出] 已经分配好的内存(大小=segment_count*data_count)，
 *                               本函数会把所有段的数据依次读到此数组中
 * @param segment_count  [输入] 总段数 (由 PreReadBigSac 获取)
 * @param data_count     [输入] 每段点数 (由 PreReadBigSac 获取)
 *
 * @return 0 成功, -1 失败
 */
int ReadBigSac(const char *big_sac,
               float *pAllData,
               unsigned segment_count,
               unsigned data_count,
               SACHEAD *pFirstHd)
{
    FILE *fp = fopen(big_sac, "rb");
    if (!fp)
    {
        fprintf(stderr, "[ReadBigSac] Error opening %s: %s\n", big_sac, strerror(errno));
        return -1;
    }

    // 1) 读取第1段头
    SACHEAD hd_first;
    if (fread(&hd_first, sizeof(SACHEAD), 1, fp) != 1)
    {
        fprintf(stderr, "[ReadBigSac] Error reading first SACHEAD.\n");
        fclose(fp);
        return -1;
    }

#ifdef BYTE_SWAP
    swab4((char *)&hd_first, sizeof(SACHEAD));
#endif

    // 将第1段头返回给调用者 (若 pFirstHd 不为NULL)
    if (pFirstHd)
    {
        *pFirstHd = hd_first;
    }

    // 检查 data_count
    unsigned first_count = (hd_first.iftype == IXY) ? hd_first.npts * 2 : hd_first.npts;
    if (first_count != data_count)
    {
        fprintf(stderr, "[ReadBigSac] First segment data_count mismatch! expected %u, got %u.\n",
                data_count, first_count);
        fclose(fp);
        return -1;
    }

    // 2) 读取第1段数据 -> pAllData 第0段区域
    size_t seg_bytes = data_count * sizeof(float);
    float *dest_first = pAllData; // 第0段
    if (fread(dest_first, seg_bytes, 1, fp) != 1)
    {
        fprintf(stderr, "[ReadBigSac] Error reading first segment data.\n");
        fclose(fp);
        return -1;
    }
#ifdef BYTE_SWAP
    swab4((char *)dest_first, seg_bytes);
#endif

    // 3) 循环读取后续 segment_count-1 段
    for (unsigned seg_idx = 1; seg_idx < segment_count; seg_idx++)
    {
        // 跳过头
        SACHEAD hd_temp;
        if (fread(&hd_temp, sizeof(SACHEAD), 1, fp) != 1)
        {
            fprintf(stderr, "[ReadBigSac] Early EOF or read error at segment #%u.\n", seg_idx + 1);
            fclose(fp);
            return -1;
        }

#ifdef BYTE_SWAP
        swab4((char *)&hd_temp, sizeof(SACHEAD));
#endif

        // 检查
        unsigned tmp_count = (hd_temp.iftype == IXY) ? hd_temp.npts * 2 : hd_temp.npts;
        if (tmp_count != data_count)
        {
            fprintf(stderr, "[ReadBigSac] Segment #%u mismatch data_count. expected %u, got %u.\n",
                    seg_idx + 1, data_count, tmp_count);
            fclose(fp);
            return -1;
        }

        // 读数据到 pAllData 对应段
        float *dest_ptr = pAllData + seg_idx * data_count;
        if (fread(dest_ptr, seg_bytes, 1, fp) != 1)
        {
            fprintf(stderr, "[ReadBigSac] Error reading data for segment #%u.\n", seg_idx + 1);
            fclose(fp);
            return -1;
        }
#ifdef BYTE_SWAP
        swab4((char *)dest_ptr, seg_bytes);
#endif
    }

    fclose(fp);
    return 0; // success
}
