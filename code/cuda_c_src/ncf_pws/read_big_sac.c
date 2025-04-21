#include "read_big_sac.h"

static unsigned get_data_count(const SACHEAD *hd)
{
    unsigned data_count = hd->npts;
    if (hd->iftype == IXY)
    {
        data_count = hd->npts * 2; // SAC 的复数/矢量类型
    }
    return data_count;
}

/**
 * @brief PreReadBigSac: 仅用于扫描 big_sac 文件，得到段数与每段的数据点数
 *
 * @param big_sac         [输入] big_sac 文件
 * @param pSegmentCount   [输出] 段数
 * @param pDataCount      [输出] 每段数据点数(由第一段 SAC 头获得)
 *
 * @return 0 表示成功, -1 表示失败
 */
int PreReadBigSac(const char *big_sac,
                  unsigned *pSegmentCount,
                  unsigned *pDataCount)
{
    FILE *fp = fopen(big_sac, "rb");
    if (!fp)
    {
        fprintf(stderr, "[PreReadBigSac] Error opening %s: %s\n",
                big_sac, strerror(errno));
        return -1;
    }

    // 读取第1段头
    SACHEAD hd_first;
    if (fread(&hd_first, sizeof(SACHEAD), 1, fp) != 1)
    {
        fprintf(stderr, "[PreReadBigSac] Error reading first SACHEAD.\n");
        fclose(fp);
        return -1;
    }

#ifdef BYTE_SWAP
    swab4((char *)&hd_first, sizeof(SACHEAD));
#endif

    unsigned base_data_count = get_data_count(&hd_first);
    if (base_data_count == 0)
    {
        fprintf(stderr, "[PreReadBigSac] Invalid data_count=0 in first segment.\n");
        fclose(fp);
        return -1;
    }

    // 跳过第1段的数据部分
    size_t seg_bytes = base_data_count * sizeof(float);
    if (fseek(fp, (long)seg_bytes, SEEK_CUR) != 0)
    {
        fprintf(stderr, "[PreReadBigSac] fseek error skipping first segment data.\n");
        fclose(fp);
        return -1;
    }

    // 段数计1
    unsigned seg_count = 1;

    // 循环扫描后续段
    while (1)
    {
        // 尝试读取下一段头
        SACHEAD hd_temp;
        size_t nread = fread(&hd_temp, sizeof(SACHEAD), 1, fp);
        if (nread < 1)
        {
            // 文件读不到新的头 -> 到尾或出错 -> 结束
            break;
        }

#ifdef BYTE_SWAP
        swab4((char *)&hd_temp, sizeof(SACHEAD));
#endif

        unsigned tmp_count = get_data_count(&hd_temp);
        if (tmp_count != base_data_count)
        {
            // 如果要求所有段大小一致，则报错退出
            fprintf(stderr, "[PreReadBigSac] Mismatch data_count in segment #%u. (Expected %u, got %u)\n",
                    seg_count + 1, base_data_count, tmp_count);
            fclose(fp);
            return -1;
        }
        // 跳过数据
        if (fseek(fp, (long)seg_bytes, SEEK_CUR) != 0)
        {
            fprintf(stderr, "[PreReadBigSac] fseek error skipping data for segment #%u.\n",
                    seg_count + 1);
            fclose(fp);
            return -1;
        }

        seg_count++;
    }

    // 输出
    *pSegmentCount = seg_count;    // 总共的段数
    *pDataCount = base_data_count; // 每段的数据点数
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
