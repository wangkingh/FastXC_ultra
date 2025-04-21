#ifndef NODE_UTIL_H
#define NODE_UTIL_H

#include "config.h"
#include "segspec.h"
#include "sac.h"
#include "complex.h"
#include <stdio.h>

/**
 * @file node_util.h
 * @brief 分层管理大规模“源-台”数据的头文件示例。
 *
 * 1. FilePaths：存放多个文件路径。
 * 2. TimeData：记录基本时间信息。
 * 3. PairBatchManager：专门管理一批（Batch）的“源-台”配对元信息，特别是相对索引、批次范围等。
 * 4. PairNodeData：存放每个节点（配对）的几何/时间等具体属性，比如经纬度、距离、有效性等。
 *
 * 这样实现“批次管理”与“数据属性”分层：BatchManager 只管哪些节点属于本批、索引区间在哪儿；
 * 而 PairNodeData 保持对各节点的物理属性描述，Filter 和后续逻辑可在此基础上填充/计算。
 */

/**
 * @struct FilePaths
 * @brief 存储多个文件路径的数组结构
 */

typedef struct FilePaths
{
    char **paths;
    int count;
} FilePaths;

typedef struct
{
    int year;        ///< 年份
    int day_of_year; ///< 年积日（001~366）
    int hour;        ///< 小时
    int minute;      ///< 分钟
    // 需要更精细时可以扩展到秒、毫秒等
} TimeData;

/**
 * @struct PairNode
 * @brief 链表节点结构，每个节点代表一对“源-台”配对，保存各种几何/时间等属性。
 *
 * 你可以根据需要调整/增加字段，比如更多的时间分辨率，跳过 step 的标志等等。
 */
typedef struct PairNode
{
    // ---------------------------
    //   1. 与批次索引相关
    // ---------------------------
    size_t source_relative_idx;  ///< 相对于本批 (src_start_idx) 的源索引
    size_t station_relative_idx; ///< 相对于本批 (sta_start_idx) 的台索引

    // ---------------------------
    //   2. 地理位置/几何信息
    // ---------------------------
    float station_lat; ///< 台站纬度
    float station_lon; ///< 台站经度
    float source_lat;  ///< 源(虚拟源)纬度
    float source_lon;  ///< 源(虚拟源)经度

    float great_circle_dist; ///< 大圆距离
    float azimuth;           ///< 方位角
    float back_azimuth;      ///< 反方位角
    float linear_distance;   ///< 线性距离或投影距离

    //   3. 时间信息
    TimeData time_info; ///< 时间信息 (年、日、时、分等)

    //   4. 有效性等
    int valid_flag; ///< 是否可用 (1=有效, 0=无效)

    //   5. 每一步的有效性
    int *step_valid_flag; ///< 每一步的有效性 (1=有效, 0=无效)
} PairNode;

/**
 * @struct PairBatchManager
 * @brief 管理单个批次（Batch）内的源台配对信息，主要是索引和批次边界，不包含物理/几何属性。
 *
 * 在大规模源台列表中，我们按一定大小 (max_sta_num) 把源和台都分成若干块，然后两两组合出
 * 许多批次，每个批次用 PairBatchManager 来记录：
 *   - 该批次在源列表、台列表中的起止索引 (src_start_idx, src_end_idx, sta_start_idx, sta_end_idx)
 *   - 批内的相对索引 (source_relative_idx, station_relative_idx)，用于快速匹配本批数据
 *   - 以及本批节点总数 (pair_count)、是否单阵列 (is_single_array) 等标志。
 *
 * 实际的“几何属性”或“时间属性”等，可以放在 PairNodeData 中，以保持职责单一。
 */
typedef struct PairBatchManager
{
    // ---------------------------
    //   1. 在全局列表中的范围
    // ---------------------------
    size_t src_start_idx; ///< 源列表在本批次中的起始索引 (左闭)
    size_t src_end_idx;   ///< 源列表在本批次中的结束索引+1 (右开)
    size_t sta_start_idx; ///< 台列表在本批次中的起始索引 (左闭)
    size_t sta_end_idx;   ///< 台列表在本批次中的结束索引+1 (右开)

    // ---------------------------
    //   2. 基本批次信息
    // ---------------------------
    size_t node_count;   ///< 该批次中节点总数 (链表中的 PairNode 数量)
    int is_single_array; ///< 是否单台阵 (1=单, 0=多)

    // ---------------------------
    //   3. 链表
    // ---------------------------
    PairNode *pair_node_array; ///< 指向本批次中所有“源-台”节点的数组
} PairBatchManager;

#endif