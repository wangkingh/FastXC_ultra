#include "par_filter_nodes.h"

ThreadPoolFilter *create_threadpool_filter_nodes(size_t num_threads)
{
    ThreadPoolFilter *pool = (ThreadPoolFilter *)malloc(sizeof(ThreadPoolFilter));
    if (!pool)
    {
        fprintf(stderr, "Memory allocation failed for ThreadPoolFilter\n");
        return NULL;
    }
    pool->threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    pool->tinfo = (thread_info_filter *)malloc(num_threads * sizeof(thread_info_filter));
    if (!pool->threads || !pool->tinfo)
    {
        free(pool->threads);
        free(pool->tinfo);
        free(pool);
        fprintf(stderr, "Memory allocation failed for threads or thread info [create_threadpool_filter_nodes]\n");
        return NULL;
    }
    pool->num_threads = num_threads;
    return pool;
}

void destroy_threadpool_filter_nodes(ThreadPoolFilter *pool)
{
    if (!pool)
        return;
    free(pool->threads);
    free(pool->tinfo);
    free(pool);
}

const char *get_basename(const char *path)
{
    const char *p = strrchr(path, '/');
    if (p == NULL)
    {
        return path;
    }
    else
    {
        return p + 1; // 跳过 '/'
    }
}

static void *filter_nodes(void *arg)
{
    thread_info_filter *tinfo = (thread_info_filter *)arg;
    PairBatchManager *batch_mgr = tinfo->batch_mgr;
    FilterCriteria *crit = tinfo->criteria;

    FilePaths *src_paths = tinfo->srcFileList;
    FilePaths *sta_paths = tinfo->staFileList;

    // 先分配临时 SEGSPEC，用于读取源与台的头信息
    SEGSPEC *phd_src = NULL;
    SEGSPEC *phd_sta = NULL;
    CpuMalloc((void **)&phd_src, sizeof(SEGSPEC));
    CpuMalloc((void **)&phd_sta, sizeof(SEGSPEC));

    // 取出批次在全局列表中的 start
    size_t srcfile_start_idx = batch_mgr->src_start_idx;
    size_t stafile_start_idx = batch_mgr->sta_start_idx;

    // 对节点数组进行遍历
    PairNode *node_array = batch_mgr->pair_node_array;
    for (size_t i = tinfo->start; i < tinfo->end; i++)
    {
        PairNode *node = &node_array[i];

        // 全局索引
        size_t global_src_idx = srcfile_start_idx + node->source_relative_idx;
        size_t global_sta_idx = stafile_start_idx + node->station_relative_idx;

        // 读取源和台站的 SEGSPEC 头
        if (read_spechead(src_paths->paths[global_src_idx], phd_src) == -1)
        {
            node->valid_flag = 0; // 读头失败 -> 标记无效
            continue;
        }
        if (read_spechead(sta_paths->paths[global_sta_idx], phd_sta) == -1)
        {
            node->valid_flag = 0; // 读头失败 -> 标记无效
            continue;
        }

        // 解析时间信息
        const char *src_fullpath = src_paths->paths[global_src_idx];
        const char *src_basename = get_basename(src_fullpath);
        int year = 0, jday = 0, hhmm = 0;
        sscanf(src_basename,
               "%*[^.].%*[^.].%4d.%3d.%4d.%*[^.].%*s",
               &year, &jday, &hhmm);
        int hour = hhmm / 100;   // 比如2359 -> 小时=23
        int minute = hhmm % 100; // 分钟=59
        node->time_info.year = year;
        node->time_info.day_of_year = jday;
        node->time_info.hour = hour;
        node->time_info.minute = minute;

        // 获取地理坐标
        double evla = phd_src->stla;
        double evlo = phd_src->stlo;
        double stla = phd_sta->stla;
        double stlo = phd_sta->stlo;

        // 计算距离和方位角
        double tempGcarc, tempAz, tempBaz, tempDist;
        distkm_az_baz_Rudoe(evlo, evla, stlo, stla, &tempGcarc, &tempAz, &tempBaz, &tempDist);

        // 填充 node
        node->source_lat = (float)evla;
        node->source_lon = (float)evlo;
        node->station_lat = (float)stla;
        node->station_lon = (float)stlo;
        node->great_circle_dist = (float)tempGcarc;
        node->azimuth = (float)tempAz;
        node->back_azimuth = (float)tempBaz;
        node->linear_distance = (float)tempDist;

        // 根据 FilterCriteria 做综合筛选
        int pass = 1;
        // 距离
        if (tempDist < crit->dist_min || tempDist > crit->dist_max)
        {
            pass = 0;
        }
        // 方位角
        if ((tempAz < crit->az_min || tempAz > crit->az_max) &&
            (tempBaz < crit->az_min || tempBaz > crit->az_max))
        {
            pass = 0;
        }

        node->valid_flag = pass; // 1=OK, 0=Not OK
    }

    // 释放临时资源
    CpuFree((void **)&phd_src);
    CpuFree((void **)&phd_sta);

    return NULL;
}

int FilterNodeParallel(PairBatchManager *batch_mgr,
                       FilePaths *src_paths, FilePaths *sta_paths,
                       ThreadPoolFilter *pool,
                       FilterCriteria *criteria)
{
    size_t total_nodes = batch_mgr->node_count;
    size_t nodes_per_thread = total_nodes / pool->num_threads;
    size_t remainder = total_nodes % pool->num_threads;

    size_t start = 0;
    for (int i = 0; i < pool->num_threads; i++)
    {
        pool->tinfo[i].batch_mgr = batch_mgr;
        pool->tinfo[i].srcFileList = src_paths;
        pool->tinfo[i].staFileList = sta_paths;
        pool->tinfo[i].criteria = criteria;

        pool->tinfo[i].start = start;
        pool->tinfo[i].end = start + nodes_per_thread + (i < remainder ? 1 : 0);

        start = pool->tinfo[i].end;
        if (pthread_create(&pool->threads[i], NULL, filter_nodes, &pool->tinfo[i]))
        {
            fprintf(stderr, "Error creating thread\n");
            return -1;
        }
    }

    // join
    for (int i = 0; i < pool->num_threads; i++)
    {
        if (pthread_join(pool->threads[i], NULL))
        {
            fprintf(stderr, "Error joining thread\n");
            return -1;
        }
    }
    return 0;
}

void CompressBatchManager(PairBatchManager *batch_mgr)
{
    PairNode *node_array = batch_mgr->pair_node_array;
    size_t write_pos = 0;
    for (size_t i = 0; i < batch_mgr->node_count; i++)
    {
        if (node_array[i].valid_flag == 1)
        {
            // 若 i != write_pos，则把 node_array[i] 挪到前面
            if (i != write_pos)
            {
                node_array[write_pos] = node_array[i];
            }
            write_pos++;
        }
    }
    batch_mgr->node_count = write_pos;
}
