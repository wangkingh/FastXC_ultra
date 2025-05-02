#include <stdio.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <libgen.h>
#include <string.h>
#include <cuComplex.h>
#include "cuda.util.cuh"
#include "cuda.pws_util.cuh"
#include "cuda.estimate_batch.cuh"
#include "cuda.stransform.cuh"
#include <stdlib.h>
extern "C"
{
#include "sac.h"
#include "prestack.h"
#include "big_sacio.h"
#include "arguproc.h"
#include "util.h"
#include "gen_ncf_path.h"
}

int main(int argc, char **argv)
{
    // ======================== 解析输入参数 ===================================
    ARGUTYPE argument;
    ArgumentProcess(argc, argv, &argument);
    char *big_sac = argument.big_sac;
    char *stack_dir = argument.stack_dir;
    size_t gpu_id = argument.gpu_id;
    int save_linear = argument.save_linear;
    int save_pws = argument.save_pws;
    int save_tfpws = argument.save_tfpws;
    int gpu_task_num = argument.gpu_task_num;
    int sub_stack_size = argument.sub_stack_size;
    char *src_info_file = argument.src_info_file; /* 新增：源信息过滤文件，可能为 NULL */

    // 分配写出文件名字符串的空间
    char src_file_name[MAXNAME];
    char net_pair[64];
    char sta_pair[64];
    char cmp_pair[64];
    char ccf_name[MAXNAME];
    char ccf_dir[MAXLINE];
    char ccf_path[2 * MAXLINE];
    char suffix[8];

    /* ------------------------------------------------------------ */
    /* 1) Pre-scan Big-SAC */
    /* ------------------------------------------------------------ */
    unsigned num_segments = 0; /* total segments */
    unsigned nsamples = 0;     /* samples per segment */
    OffItem *offs;

    if (PreScanBigSac(big_sac, &num_segments, &nsamples, &offs) != 0)
    {
        fprintf(stderr, "scan failed\n");
        return 1;
    }

    printf("Big-SAC contains %u segments, each with %u samples.\n",
           num_segments, nsamples);

    size_t group_sz = (sub_stack_size < 1) ? 1 : (size_t)sub_stack_size;
    unsigned ngroups = (num_segments + group_sz - 1) / group_sz;
    size_t bytes_outmat = ngroups * nsamples * sizeof(float); /* 压缩后 all_data */

    /* ------------------------------------------------------------ */
    /*  2) Allocate buffers for pre-stack                              */
    /* ------------------------------------------------------------ */
    float *prestack_data = NULL;  /* output matrix after (sub)stack   */
    float *h_linear_stack = NULL; /* global linear-sum accumulator    */

    prestack_data = (float *)calloc(bytes_outmat / sizeof(float), sizeof(float));
    if (!prestack_data)
    {
        fprintf(stderr, "calloc(prestack_data %.2f MiB) failed\n",
                bytes_outmat / (1024.0 * 1024.0));
        return -1;
    }

    /* ------------------------------------------------------------
     *  3) build uniform groups  (不复制 OffItem)
     * -----------------------------------------------------------*/
    OffGroup *groups;                       /* 目录表 */
    if (group_offsets_uniform(offs,         /* offset 表   */
                              num_segments, /* 条目数      */
                              group_sz,     /* 组大小      */
                              &groups, &ngroups) != 0)
    {
        fprintf(stderr, "grouping failed\n");
        free(offs);
        return 1;
    }

    /* ------------------------------------------------------------
     *  4) 读取文件 → 填充预叠加矩阵 prestack_data,获取第一个head
     * -----------------------------------------------------------*/
    FILE *fp = fopen(big_sac, "rb");
    if (!fp)
    {
        perror(big_sac);
        free_groups(groups);
        free(offs);
        return 1;
    }

    SACHEAD first_hd;
    if (fseeko(fp, offs[0].hdr_off, SEEK_SET) != 0 ||
        fread(&first_hd, sizeof(SACHEAD), 1, fp) != 1)
    {
        perror("read first hdr");
        fclose(fp);
        free_groups(groups);
        free(offs);
        return 1;
    }
#ifdef BYTE_SWAP
    SWAP4(&first_hd, sizeof(SACHEAD) / 4);
#endif

#ifdef BYTE_SWAP
    int doswap = 1;
#else
    int doswap = 0;
#endif

    if (fill_prestack_matrix(fp,
                             groups, ngroups,
                             nsamples,
                             doswap,
                             prestack_data) != 0)
    {
        fprintf(stderr, "fill_prestack_matrix failed\n");
        fclose(fp);
        free_groups(groups);
        free(offs);
        return 1;
    }
    fclose(fp);          /* 文件读取完毕，可关闭        */
    free_groups(groups); /* 目录表释放                  */
    free(offs);          /* offset 表也不再需要         */

    /* ------------------------------------------------------------*/
    // 5) 线性叠加
    /* ------------------------------------------------------------*/
    h_linear_stack = (float *)calloc(nsamples, sizeof(float));
    if (!h_linear_stack)
    {
        fprintf(stderr, "calloc h_lienar_stack failed\n");
        return -1;
    }
    for (size_t i = 0; i < ngroups; i++)
    {
        for (size_t j = 0; j < nsamples; j++)
        {
            h_linear_stack[j] += prestack_data[i * nsamples + j] / (float)ngroups;
        }
    }

    // 写入头部信息
    SACHEAD ncf_hd = first_hd;
    ncf_hd.npts = (int)nsamples;
    ncf_hd.iftype = ITIME;
    ncf_hd.user0 = (float)ngroups;
    ncf_hd.nzyear = 2010;
    ncf_hd.nzjday = 214;
    ncf_hd.nzhour = 0;
    ncf_hd.nzmin = 0;
    ncf_hd.nzmsec = 0;

    // 获取文件名信息
    strncpy(src_file_name, basename(big_sac), MAXNAME);
    SplitFileName(src_file_name, ".", net_pair, sta_pair, cmp_pair, suffix);

    // =================== 写 出 线 性 叠 加 数 据======================
    if (save_linear == 1)
    {
        snprintf(ccf_dir, sizeof(ccf_dir), "%s/linear/%s.%s/", stack_dir, net_pair, sta_pair);
        snprintf(ccf_name, MAXLINE, "%s.%s.%s.linear.sac", net_pair, sta_pair, cmp_pair);
        CreateDir(ccf_dir);
        snprintf(ccf_path, 2 * MAXLINE, "%s/%s", ccf_dir, ccf_name);
        write_sac(ccf_path, ncf_hd, h_linear_stack); // 使用新的文件名写文件
    }

    // 如果没有加权叠加，直接返回
    if (save_pws == 0 && save_tfpws == 0)
    {
        free(prestack_data);
        free(h_linear_stack);
        return 0;
    }

    // ============================== 加权叠加预处理 =========================================
    CUDACHECK(cudaSetDevice(gpu_id));

    // 定义网格
    dim3 dimGrid_1D, dimBlock_1D;
    dim3 dimGrid_2D, dimBlock_2D;
    dim3 dimGrid_3D, dimBlock_3D;

    // 拷贝线性叠加结果到显存
    float *d_linear_stack = NULL; // 保存线性叠加结果
    GpuMalloc((void **)&d_linear_stack, nsamples * sizeof(float));
    CUDACHECK(cudaMemcpy(d_linear_stack, h_linear_stack, nsamples * sizeof(float), cudaMemcpyHostToDevice));
    CpuFree((void **)&h_linear_stack); // 释放主机内存

    // 拷贝所有的道数据到显存
    float *d_ncf_buffer_all = NULL; // 用于存储所有的道数据
    GpuMalloc((void **)&d_ncf_buffer_all, ngroups * nsamples * sizeof(float));
    CUDACHECK(cudaMemcpy(d_ncf_buffer_all, prestack_data, ngroups * nsamples * sizeof(float), cudaMemcpyHostToDevice));
    CpuFree((void **)&prestack_data); // 释放主机内存

    // ======== 0 希尔伯特变换，无论是pws还是tf-pws,都需要进行希尔伯特变换将数组变为解析信号========
    // 创建给希尔伯特变换的FFT计划,执行正变换,将每一道数据转换为频率
    cuComplex *d_spectrum = NULL; // 输入时间序列对应的频谱,用于希尔伯特变换
    GpuMalloc((void **)&d_spectrum, ngroups * nsamples * sizeof(cufftComplex));
    int rank_hilb = 1;
    int n_hilb[1] = {(int)nsamples};
    int inembed[1] = {(int)nsamples};
    int onembed[1] = {(int)nsamples};
    int istride = 1;
    int ostride = 1;
    int idist = (int)nsamples;
    int odist = (int)nsamples;

    cufftHandle plan_fwd;
    CufftPlanAlloc(&plan_fwd, rank_hilb, n_hilb, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, ngroups);
    CUFFTCHECK(cufftExecR2C(plan_fwd, (cufftReal *)d_ncf_buffer_all, (cufftComplex *)d_spectrum));
    DimCompute2D(&dimGrid_2D, &dimBlock_2D, nsamples, ngroups);
    hilbertTransformKernel<<<dimGrid_2D, dimBlock_2D>>>(d_spectrum, nsamples, ngroups);
    cufftDestroy(plan_fwd);
    GpuFree((void **)&d_ncf_buffer_all);

    // ======================1. PWS叠加, 使用蒋磊代码=======================================
    if (save_pws == 1)
    {
        float *d_pw_stack = NULL;          // 仅保留相位加权叠加结果时用到
        cuComplex *hilbert_complex = NULL; // 创建储存反变换后的解析信号的数组
        cuComplex *analyze_mean = NULL;    // 解析信号求和再求平均,原代码中的divide_mean
        float *weight = NULL;              // 存储由多道解析信号加和得到的权重
        cufftHandle plan_inv_pws;          // C2C 用于将希尔伯特频谱转变为解析信号
        CUDACHECK(cudaMalloc((void **)&hilbert_complex, ngroups * nsamples * sizeof(cuComplex)));
        CUDACHECK(cudaMalloc(&analyze_mean, nsamples * sizeof(cufftComplex)));
        CUDACHECK(cudaMalloc(&weight, nsamples * sizeof(float)));
        CUDACHECK(cudaMalloc(&d_pw_stack, nsamples * sizeof(float)));
        cufftPlanMany(&plan_inv_pws, rank_hilb, n_hilb, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, ngroups);
        DimCompute2D(&dimGrid_2D, &dimBlock_2D, nsamples, ngroups);

        // C2C 反变换, 希尔伯特频谱[d_spectrum]->解析信号[hilbert_complex]
        CUFFTCHECK(cufftExecC2C(plan_inv_pws, d_spectrum, hilbert_complex, CUFFT_INVERSE));

        // 归一化解析信号到单位圆上,nfft归一化因子用来控制序列能量
        DimCompute1D(&dimGrid_1D, &dimBlock_1D, nsamples * ngroups);
        cudaNormalizeComplex<<<dimGrid_1D, dimBlock_1D>>>(hilbert_complex, nsamples * ngroups, nsamples);

        // 计算权重并使用权重归一化
        DimCompute1D(&dimGrid_1D, &dimBlock_1D, nsamples);
        cudaMean<<<dimGrid_1D, dimBlock_1D>>>(hilbert_complex, analyze_mean, ngroups, nsamples);
        cudaMultiply<<<dimGrid_1D, dimBlock_1D>>>(d_linear_stack, analyze_mean, d_pw_stack, nsamples);

        float *pw_stack = (float *)malloc(nsamples * sizeof(float)); // 相位加权叠加结果
        CUDACHECK(cudaMemcpy(pw_stack, d_pw_stack, nsamples * sizeof(float), cudaMemcpyDeviceToHost));
        snprintf(ccf_name, MAXLINE, "%s.%s.%s.pws.sac", net_pair, sta_pair, cmp_pair);
        snprintf(ccf_dir, sizeof(ccf_dir), "%s/pws/%s.%s/", stack_dir, net_pair, sta_pair);
        snprintf(ccf_path, 2 * MAXLINE, "%s/%s", ccf_dir, ccf_name);
        CreateDir(ccf_dir);
        write_sac(ccf_path, ncf_hd, pw_stack);

        // 释放线性叠加的显卡空间
        CpuFree((void **)&pw_stack);
        GpuFree((void **)&d_pw_stack);
        GpuFree((void **)&hilbert_complex);
        GpuFree((void **)&analyze_mean);
        GpuFree((void **)&weight);
        cufftDestroy(plan_inv_pws);
    }

    // ======== 2. tf-pws 时频域相位加权叠加, 修改自 Li Guoliang 的程序 ===========
    if (save_tfpws == 1)
    {
        // 2.0 计算一些参数
        size_t num_freq_bins = nsamples / 2 + 1; // 设置调制频率数, 长度等同于Nyquist采样频率
        size_t freq_batch_size = EstimateFreqBatchSize(
            gpu_id,
            ngroups,
            nsamples,
            num_freq_bins,
            gpu_task_num,
            0.8f);

        size_t num_freq_batches = (num_freq_bins + freq_batch_size - 1) / freq_batch_size;
        float weight_order = 1; // 设置加权阶数
        float scale = 0.1;      // 设置高斯窗的宽度, 0.1是一个比较好的值, 0.2会导致频谱变得很宽

        // ========== 2.1 分配 GPU 内存 ============
        // (A) 权重矩阵: 大小 [num_freq_bins, nsamples], 复数
        cuComplex *d_tfpws_weight;
        CUDACHECK(cudaMalloc(&d_tfpws_weight, num_freq_bins * nsamples * sizeof(cuComplex)));
        CUDACHECK(cudaMemset(d_tfpws_weight, 0, num_freq_bins * nsamples * sizeof(cuComplex)));

        // (B) TF-PWS 最终输出(复数 & 实数)
        cufftComplex *d_tfpw_stack_complex;
        float *d_tfpw_stack;
        CUDACHECK(cudaMalloc((void **)&d_tfpw_stack_complex, nsamples * sizeof(cufftComplex)));
        CUDACHECK(cudaMalloc((void **)&d_tfpw_stack, nsamples * sizeof(float)));

        // (C) 叠后数据频谱 + 调制后的叠后数据 (时频表示)
        cufftComplex *d_stacked_spectrum;    // [nsamples]  (仅1道叠后数据)
        cufftComplex *d_st_stacked_spectrum; // [nsamples * num_freq_bins] (S变换结果)
        CUDACHECK(cudaMalloc((void **)&d_stacked_spectrum, nsamples * sizeof(cufftComplex)));
        CUDACHECK(cudaMalloc((void **)&d_st_stacked_spectrum, num_freq_bins * nsamples * sizeof(cufftComplex)));

        // ========== 2.2 创建 FFT 计划 ============
        // 用到 3 个 plan: fftPlanFwd_singleTrace, fftPlanInv_multiFreq, fftPlanInv_singleTrace, planinv_tfpws
        cufftHandle fftPlanFwd_singleTrace, fftPlanInv_multiFreq, fftPlanInv_singleTrace;

        // 2.2.1 fftPlanFwd_singleTrace => R2C, batch=1 (对叠后数据做R2C)
        cufftPlanMany(&fftPlanFwd_singleTrace, rank_hilb, n_hilb, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, 1);

        // 2.2.2 fftPlanInv_multiFreq => C2C, batch=num_freq_bins (对[叠后]调制频谱做逆变换)
        cufftPlanMany(&fftPlanInv_multiFreq, rank_hilb, n_hilb, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, (int)num_freq_bins);

        // 2.2.3 fftPlanInv_singleTrace => C2C, batch=1 (最后一步对加权频谱逆变换回时域)
        cufftPlanMany(&fftPlanInv_singleTrace, rank_hilb, n_hilb, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, 1);

        // ========== 2.3 对[叠后数据]做 Stockwell 变换(相当于S变换) ============
        // (A) R2C => 叠后频谱
        CUFFTCHECK(cufftExecR2C(fftPlanFwd_singleTrace, (cufftReal *)d_linear_stack, (cufftComplex *)d_stacked_spectrum));

        // (B) hilbertTransformKernel => 得到解析信号频谱 (后半部分为0)
        DimCompute2D(&dimGrid_2D, &dimBlock_2D, nsamples, 1);
        hilbertTransformKernel<<<dimGrid_2D, dimBlock_2D>>>(d_stacked_spectrum, nsamples, 1);

        // (C) gaussianModulate => 得到叠后数据的[频率调制]  (1道 => batch=1)
        DimCompute3D(&dimGrid_3D, &dimBlock_3D, nsamples, num_freq_bins, 1);
        gaussianModulateSub<<<dimGrid_3D, dimBlock_3D>>>(
            /* d_inputSpectrum       */ d_stacked_spectrum,
            /* d_modulatedSubChunk   */ d_st_stacked_spectrum,
            /* nTraces               */ 1,
            /* freqDomainLen         */ nsamples,
            /* chunkStartFreq        */ 0,
            /* chunkFreqCount        */ (int)num_freq_bins,
            /* scale                 */ scale);

        // (D) C2C 逆变换 => 中心频率-中心时间 域
        CUFFTCHECK(cufftExecC2C(fftPlanInv_multiFreq, d_st_stacked_spectrum, d_st_stacked_spectrum, CUFFT_INVERSE));

        // 销毁 fwd 和 inv_trace 这两个用完的 plan
        cufftDestroy(fftPlanFwd_singleTrace);
        cufftDestroy(fftPlanInv_multiFreq);

        // ========== 2.4 对多道数据一次性计算权重 =============
        // (A) 我们先分配一个“子频段调制”临时数组 d_st_subbatch
        //     大小： [ngroups, freq_batch_size, nsamples]
        cufftComplex *d_st_subbatch;
        CUDACHECK(cudaMalloc((void **)&d_st_subbatch, (size_t)ngroups * freq_batch_size * nsamples * sizeof(cufftComplex)));

        // (B) 还需要一个子批次 FFT 计划 => batch = ngroups*freq_batch_size
        cufftHandle fftPlanInv_subMultiTrace;
        cufftPlanMany(&fftPlanInv_subMultiTrace,
                      rank_hilb, n_hilb,
                      inembed, istride, idist,
                      onembed, ostride, odist,
                      CUFFT_C2C, (int)(ngroups * freq_batch_size));

        // (D) 频率分块循环
        for (int ichunk = 0; ichunk < num_freq_batches; ichunk++)
        {
            int f_start = ichunk * freq_batch_size;
            int sub_nfreq = ((f_start + freq_batch_size) <= (int)num_freq_bins)
                                ? freq_batch_size
                                : ((int)num_freq_bins - f_start);

            // (D1) 调用 "gaussianModulateSub<<<>>>", 仅对 [f_start..f_start+sub_nfreq) 做高斯窗调制
            //      将 d_spectrum_temp => d_st_subbatch
            //      大小: [ngroups, sub_nfreq, nsamples]
            dim3 gridG, blockG;
            DimCompute3D(&gridG, &blockG, nsamples, sub_nfreq, ngroups);
            gaussianModulateSub<<<gridG, blockG>>>(
                /* d_inputSpectrum       */ d_spectrum,
                /* d_modulatedSubChunk   */ d_st_subbatch,
                /* nTraces               */ ngroups,
                /* freqDomainLen         */ nsamples,
                /* chunkStartFreq        */ f_start,
                /* chunkFreqCount        */ sub_nfreq,
                /* scale                 */ scale);

            // (D2) 对这块做 IFFT => fftPlanInv_subMultiTrace
            cufftExecC2C(fftPlanInv_subMultiTrace, d_st_subbatch, d_st_subbatch, CUFFT_INVERSE);

            // (D3) 计算权重 => 只对子频段 [f_start..f_start+sub_nfreq) 做相位一致性统计
            //      累加到全局 d_tfpws_weight
            dim3 gridW, blockW;
            DimCompute2D(&gridW, &blockW, nsamples, sub_nfreq);
            calculateWeightSub<<<gridW, blockW>>>(
                d_st_subbatch,  // [ngroups, sub_nfreq, nsamples]
                d_tfpws_weight, // [num_freq_bins, nsamples]
                nsamples,
                ngroups,
                f_start,
                sub_nfreq);
        }

        // ========== 2.5 应用权重到叠后数据 (d_st_stacked_spectrum ==========

        // (A) Kernel applyWeight: 大小 = [num_freq_bins, nsamples]
        //     叠后数据只有1道 => d_st_stacked_spectrum的 shape ~ [num_freq_bins, nsamples]
        DimCompute2D(&dimGrid_2D, &dimBlock_2D, nsamples, num_freq_bins);

        // 对 [num_freq_bins, nsamples] 的叠后数据乘以 d_tfpws_weight
        applyWeight<<<dimGrid_2D, dimBlock_2D>>>(
            d_st_stacked_spectrum, // [num_freq_bins, nsamples]
            d_tfpws_weight,        // [num_freq_bins, nsamples]
            num_freq_bins,
            nsamples,
            weight_order);

        // (B) 将加权后的叠后数据在“时间方向”叠加 => [num_freq_bins]
        //     即把 [num_freq_bins, nsamples] 上每个 time slice 累加 => [num_freq_bins]
        CUDACHECK(cudaMemset(d_stacked_spectrum, 0, nsamples * sizeof(cufftComplex)));
        // DimCompute1D(&dimGrid_1D, &dimBlock_1D, nsamples);
        // 加和前一半的频率，只有前一半的频率有意义
        DimCompute1D(&dimGrid_1D, &dimBlock_1D, num_freq_bins);
        sumOverTimeAxisKernel<<<dimGrid_1D, dimBlock_1D>>>(
            d_st_stacked_spectrum, // d_tfAnalysis, shape=[num_freq_bins, nsamples]
            d_stacked_spectrum,    // d_outSpectrum, shape=[nsamples]
            num_freq_bins,
            nsamples);
        ;
        // (C) 逆变换 => 得到时域 TF-PWS 结果 (复数)
        CUFFTCHECK(cufftExecC2C(fftPlanInv_singleTrace, d_stacked_spectrum, d_tfpw_stack_complex, CUFFT_INVERSE));
        cufftDestroy(fftPlanInv_singleTrace);

        // (D) 提取实部 => d_tfpw_stack
        DimCompute1D(&dimGrid_1D, &dimBlock_1D, nsamples);
        extractReal<<<dimGrid_1D, dimBlock_1D>>>(d_tfpw_stack,
                                                 d_tfpw_stack_complex,
                                                 nsamples);
        // (E) 销毁临时数组
        GpuFree((void **)&d_stacked_spectrum);
        GpuFree((void **)&d_st_stacked_spectrum);
        GpuFree((void **)&d_tfpw_stack_complex);
        GpuFree((void **)&d_tfpws_weight);

        // ========== 2.6 写结果到 SAC =============
        float *tfpw_stack = (float *)malloc(nsamples * sizeof(float));
        CUDACHECK(cudaMemcpy(tfpw_stack, d_tfpw_stack,
                             nsamples * sizeof(float), cudaMemcpyDeviceToHost));
        snprintf(ccf_name, MAXLINE, "%s.%s.%s.tfpws.sac", net_pair, sta_pair, cmp_pair);
        snprintf(ccf_dir, sizeof(ccf_dir), "%s/tfpws/%s.%s/", stack_dir, net_pair, sta_pair);
        snprintf(ccf_path, 2 * MAXLINE, "%s/%s", ccf_dir, ccf_name);
        CreateDir(ccf_dir);
        write_sac(ccf_path, ncf_hd, tfpw_stack);

        free(tfpw_stack);
        GpuFree((void **)&d_tfpw_stack);
    }

    // 销毁输出数组
    GpuFree((void **)&d_linear_stack);
    GpuFree((void **)&d_spectrum);
    return 0;
}