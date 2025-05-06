#include "my_write_sac.h"

/***********************************************************
  read_sac_buffer

  Description:	read binary data from file. If succeed, it will return
                a float pointer to the data array.

  Author:	Lupei Zhu

  Modified by wang Jingxi

  Arguments:	const char *name 	file name
                int npts

  Return:	float pointer to the data array, NULL if failed

  Modify history:
        09/20/93	Lupei Zhu	Initial coding
        12/05/96	Lupei Zhu	adding error handling
        12/06/96	Lupei Zhu	swap byte-order on PC
************************************************************/

float *read_sac_buffer(const char *name, SACHEAD *sac_hd, float *buffer)
{
    FILE *strm;
    unsigned sz;

    if ((strm = fopen(name, "rb")) == NULL)
    {
        fprintf(stderr, "Unable to open %s\n", name);
        return NULL;
    }

    if (fseek(strm, sizeof(SACHEAD), SEEK_SET) != 0)
    {
        fprintf(stderr, "Error in skipping SAC header %s\n", name);
        fclose(strm);
        return NULL;
    }

#ifdef BYTE_SWAP
    swab4((char *)hd, HD_SIZE);
#endif

    sz = sac_hd->npts * sizeof(float);

    if (fread((char *)buffer, sz, 1, strm) != 1)
    {
        fprintf(stderr, "Error in reading SAC data %s\n", name);
        return NULL;
    }

    fclose(strm);

#ifdef BYTE_SWAP
    swab4((char *)buffer, sz);
#endif
    return buffer;
}

// 简单检查文件是否存在的函数
static int file_exists(const char *name)
{
    struct stat st;
    return (stat(name, &st) == 0);
}

// 标准写入函数：为给定文件新建并写入SAC头和数据（传统模式的核心逻辑）
static int normal_write_sac(const char *name, SACHEAD hd, const float *ar)
{
    FILE *strm = NULL;
    unsigned sz = hd.npts * sizeof(float);
    if (hd.iftype == IXY)
        sz *= 2;

    float *data = (float *)malloc(sz);
    if (!data)
    {
        fprintf(stderr, "Error in allocating memory for writing %s\n", name);
        return -1;
    }

    if (!memcpy(data, ar, sz))
    {
        fprintf(stderr, "Error in copying data for writing %s\n", name);
        free(data);
        return -1;
    }

#ifdef BYTE_SWAP
    swab4((char *)data, sz);
    swab4((char *)&hd, HD_SIZE);
#endif

    strm = fopen(name, "w");
    if (!strm)
    {
        fprintf(stderr, "Error in opening file for writing %s\n", name);
        free(data);
        return -1;
    }

    if (fwrite(&hd, sizeof(SACHEAD), 1, strm) != 1)
    {
        fprintf(stderr, "Error in writing SAC header for %s\n", name);
        fclose(strm);
        free(data);
        return -1;
    }

    if (fwrite(data, sz, 1, strm) != 1)
    {
        fprintf(stderr, "Error in writing SAC data for %s\n", name);
        fclose(strm);
        free(data);
        return -1;
    }

    fclose(strm);
    free(data);
    return 0;
}

// 追加模式写入函数：如果文件存在，就以追加模式打开并写入数据尾部；否则回退到normal_write_sac
static int append_write_sac(const char *name, SACHEAD hd, const float *ar)
{
    if (!file_exists(name))
    {
        // 文件不存在，则回退到传统模式创建新文件
        return normal_write_sac(name, hd, ar);
    }

    // 文件存在，尝试打开并追加写入, 大文件格式是多段SAC数据串联，头+数据+头+数据...依次追加
    FILE *strm = fopen(name, "ab");
    if (!strm)
    {
        fprintf(stderr, "Error opening file in append mode %s\n", name);
        return -1;
    }

    unsigned sz = hd.npts * sizeof(float);
    if (hd.iftype == IXY)
        sz *= 2;
    float *data = (float *)malloc(sz);
    if (!data)
    {
        fprintf(stderr, "Memory error in append_write_sac\n");
        fclose(strm);
        return -1;
    }

    if (!memcpy(data, ar, sz))
    {
        fprintf(stderr, "Memcpy error in append_write_sac\n");
        free(data);
        fclose(strm);
        return -1;
    }

#ifdef BYTE_SWAP
    swab4((char *)data, sz);
    swab4((char *)&hd, HD_SIZE);
#endif

    // 写入新的头和数据块到文件末尾
    if (fwrite(&hd, sizeof(SACHEAD), 1, strm) != 1)
    {
        fprintf(stderr, "Error writing header in append mode %s\n", name);
        free(data);
        fclose(strm);
        return -1;
    }

    if (fwrite(data, sz, 1, strm) != 1)
    {
        fprintf(stderr, "Error writing data in append mode %s\n", name);
        free(data);
        fclose(strm);
        return -1;
    }

    free(data);
    fclose(strm);
    return 0;
}

// 叠加模式写入函数：如果文件已存在，则读出已有数据叠加上新数据，再写回；
// 否则创建新文件(回退到normal_write_sac)。
static int aggregate_write_sac(const char *name, SACHEAD hd, const float *ar)
{
    if (!file_exists(name))
    {
        // 文件不存在则采用传统模式创建基础文件
        return normal_write_sac(name, hd, ar);
    }

    // 文件已存在，先读取旧头
    SACHEAD old_hd;
    if (read_sachead(name, &old_hd) != 0)
    {
        fprintf(stderr, "Failed to read old header from %s\n", name);
        return -1;
    }

    // ========== 新增: 检查文件大小是否为"单段SAC" ==========
    // 1. 取文件实际大小
    long long file_size = 0;
    {
        FILE *fcheck = fopen(name, "rb");
        if (!fcheck)
        {
            fprintf(stderr, "Cannot open file for size check: %s\n", name);
            return -1;
        }
        if (fseek(fcheck, 0, SEEK_END) != 0)
        {
            fclose(fcheck);
            fprintf(stderr, "fseek() failed for size check: %s\n", name);
            return -1;
        }
        file_size = ftell(fcheck); // 获取文件字节数
        fclose(fcheck);
    }

    // 2. 根据旧头，计算"标准单段"的期望大小
    //    sizeof(SACHEAD) + npts * ( sizeof(float) [or 2*sizeof(float) if IXY ] )
    long long expected_size = sizeof(SACHEAD);
    if (old_hd.iftype == IXY)
    {
        // 对于 IXY 类型：一个采样点由2个float表示 => npts * 2 * sizeof(float)
        expected_size += old_hd.npts * 2 * sizeof(float);
    }
    else
    {
        // 普通单分量 => npts * sizeof(float)
        expected_size += old_hd.npts * sizeof(float);
    }

    // 3. 如果文件实际大小 != 期望大小，则认为它不是标准的单段SAC文件（可能是多段APPEND形成的）
    if (file_size != expected_size)
    {
        fprintf(stderr,
                "File size (%lld) not match single-segment SAC expected size (%lld). "
                "Skip aggregation.\n",
                file_size, expected_size);

        // 这里可以根据你需求决定返回什么，或直接返回 0/–1 都行
        // 例如返回 -1 表示“跳过，不再叠加”
        return -1;
    }

    // 文件已存在，读取旧数据
    float *old_data = (float *)malloc(hd.npts * sizeof(float));
    if (!old_data)
    {
        fprintf(stderr, "Memory allocation error in aggregate mode for %s\n", name);
        return -1;
    }

    // kuser27 标记了叠加数量，叠加模式下要+1
    old_hd.unused27 = old_hd.unused27 + 1;
    hd.unused27 = old_hd.unused27;
    if (read_sac_buffer(name, &old_hd, old_data) == NULL)
    {
        fprintf(stderr, "Failed to read existing SAC data in aggregate mode for %s\n", name);
        free(old_data);
        return -1;
    }

    // 假设hd.npts == old_hd.npts并且格式一致，如果不一致需要额外检查
    // 执行叠加逻辑 (示例：简单相加)
    for (int i = 0; i < hd.npts; i++)
    {
        old_data[i] = old_data[i] + ar[i];
    }

#ifdef BYTE_SWAP
    swab4((char *)&hd, HD_SIZE);
    swab4((char *)old_data, hd.npts * sizeof(float));
#endif

    // 以"r+b"模式打开文件覆盖写入
    FILE *strm = fopen(name, "r+b");
    if (!strm)
    {
        fprintf(stderr, "Error opening file in aggregate mode %s\n", name);
        free(old_data);
        return -1;
    }

    // 回到文件头写入新的头和叠加后数据
    if (fseek(strm, 0, SEEK_SET) != 0)
    {
        fprintf(stderr, "Error seeking to start in aggregate mode %s\n", name);
        free(old_data);
        fclose(strm);
        return -1;
    }

    if (fwrite(&hd, sizeof(SACHEAD), 1, strm) != 1)
    {
        fprintf(stderr, "Error rewriting header in aggregate mode %s\n", name);
        free(old_data);
        fclose(strm);
        return -1;
    }

    unsigned sz = hd.npts * sizeof(float);
    if (hd.iftype == IXY)
        sz *= 2;

    if (fwrite(old_data, sz, 1, strm) != 1)
    {
        fprintf(stderr, "Error rewriting data in aggregate mode %s\n", name);
        free(old_data);
        fclose(strm);
        return -1;
    }

    free(old_data);
    fclose(strm);

    return 0;
}

// 主函数：根据mode选择不同策略
int my_write_sac(const char *name, SACHEAD hd, const float *ar, int write_mode)
{
    // display write mode
    // printf("Write mode: %d\n", write_mode);
    switch (write_mode)
    {
    case MODE_APPEND:
        // print name
        printf("file name is %s\n", name);
        return append_write_sac(name, hd, ar);
    case MODE_AGGREGATE:
        return aggregate_write_sac(name, hd, ar);
    default:
        fprintf(stderr, "Unknown mode %d for my_write_sac\n", write_mode);
        return -1;
    }
}