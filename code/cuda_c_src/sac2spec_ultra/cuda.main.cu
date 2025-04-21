/* last updated by wangjx@20250421 */

#include "cuda.processing.cuh"
#include "cuda.util.cuh"

extern "C"
{
#include "arguproc.h"
#include "design_filter_response.h"
#include "in_out_node.h"
#include "cal_nseg.h"
#include "read_filelist.h"
#include "par_rw_data.h"
#include "sac.h"
#include "util.h"
}

int main(int argc, char **argv)
{
    // Parsing arguments
    ARGUTYPE argument;
    ArgumentProcess(argc, argv, &argument);
    PathNode *pInFileList = readPathList(argument.sac_lst);   // Read in sac path list file
    PathNode *pOutFileList = readPathList(argument.spec_lst); // Read in spec output path list file

    createDirectories(pOutFileList); // Create output_dir for .segspec files

    FilePathArray InPaths = PathList2Array(pInFileList); // Turn file chain input list chain into array
    FilePathArray OutPaths = PathList2Array(pOutFileList);

    int num_ch = argument.num_ch; // Get the number of channels

    // Parsing Whiten and Normalization Type
    int wh_before = 0, wh_after = 0, do_runabs_mf = 0, do_runabs = 0, do_onebit = 0;
    switch (argument.whitenType)
    {
    case 0:
        wh_before = 0, wh_after = 0;
        break;
    case 1:
        wh_before = 1;
        break;
    case 2:
        wh_after = 1;
        break;
    case 3:
        wh_before = 1, wh_after = 1;
        break;
    default:
        printf("Invalid value for bandwhiten\n");
    }

    switch (argument.normalizeType)
    {
    case 0:
        do_runabs_mf = 0, do_onebit = 0, do_runabs = 0;
        break;
    case 1:
        do_runabs_mf = 1;
        break;
    case 2:
        do_onebit = 1;
        break;
    case 3:
        do_runabs = 1;
        break;
    default:
        printf("Invalid value for normalization\n");
    }

    size_t nValid_sacnum = InPaths.count;         // Count the number of input files
    size_t nValid_batch = nValid_sacnum / num_ch; // Count the number of input files/channels

    size_t gpu_id = argument.gpu_id; // Set the gpu_id
    cudaSetDevice(gpu_id);

    SACHEAD sachd;
    if (read_sachead(pInFileList->path, &sachd) != 0)
    {
        fprintf(stderr, "ERROR reading first SACHEAD\n");
        exit(EXIT_FAILURE);
    }
    int npts = sachd.npts; // Read npts and delta form the file file of pInFileList
    float delta = sachd.delta;

    int segment_pts_1x = cal_segment_length(argument.seglen, npts, delta); // Calculate the number of points of sement

    int shift_length_pts = (int)(argument.segshift / delta + 0.5f);
    if (shift_length_pts < 1)
        shift_length_pts = segment_pts_1x;
    int nstep = 0;
    if (segment_pts_1x <= npts)
    {
        // (npts - window_length_pts) / shift_length_pts + 1
        nstep = (npts - segment_pts_1x) / shift_length_pts + 1;
        if (nstep < 1)
            nstep = 1;
    }
    else
    {
        nstep = 1; // if window_length_pts > npts, only one segment
    }
    printf("nstep = %d\n", nstep);

    int segment_pts_2x = segment_pts_1x * 2; // Create segment_pts_2x for zero padding data
    int nspec_output = segment_pts_2x / 2 + 1;

    float df_1x = 1.0 / (segment_pts_1x * delta);
    float df_2x = 1.0 / (segment_pts_2x * delta);

    // Parse frequcny band, calculate idx of corner and cutoff frequency
    float freq_low = argument.freq_low;
    float freq_high = argument.freq_high;
    int f_idx1 = int(freq_low * 0.667 / df_1x);
    int f_idx2 = int(freq_low / df_1x);
    int f_idx3 = int(freq_high / df_1x);
    int f_idx4 = int(freq_high * 1.333 / df_1x);

    int filter_count = 0;
    ButterworthFilter *filter = readButterworthFilters(argument.filter_file, &filter_count);     // read in filter file
    FilterResp *myResp = processButterworthFilters(filter, filter_count, df_2x, segment_pts_2x); // Calculate filter f domain response

    // Parsing skip_steps
    int *skip_steps = argument.skip_steps;
    int skip_step_count = argument.skip_step_count;
    int nstep_valid = nstep;

    int skip_flags[nstep]; // Mark which steps should be skipped
    memset(skip_flags, false, sizeof(skip_flags));

    for (int i = 0; i < skip_step_count; ++i)
    {
        int skip_step = skip_steps[i];
        if (skip_step >= 0 && skip_step < nstep)
        {
            skip_flags[skip_step] = true;
            printf("Step [no. %d] will be skipped \n", skip_step);
        }
    }

    nstep_valid = 0; // Calculate validated steps
    for (int i = 0; i < nstep; ++i)
    {
        if (!skip_flags[i])
        {
            ++nstep_valid;
        }
    }

    // ********* Calculate CPU memory *********** //
    float *h_sacdata = NULL;
    complex *h_spectrum = NULL;

    InOutNode *pInOutList = NULL;

    size_t unit_sacdata_size = npts * sizeof(float);                          // input sac data
    size_t unit_spectrum_size = nstep_valid * nspec_output * sizeof(complex); // output total spectrum
    size_t unit_InOutNode_size = sizeof(InOutNode);                           // contain head/path/data
    size_t unit_thread_write_size = sizeof(thread_info_write);                // contain thread_info_write
    size_t unit_thread_read_size = sizeof(thread_info_read);                  // contain thread_info_read
    // print unit size of each part in MB
    printf("unit_sacdata_size = %ld MB\n", unit_sacdata_size / 1024 / 1024);
    printf("unit_spectrum_size = %ld MB\n", unit_spectrum_size / 1024 / 1024);
    printf("unit_InOutNode_size = %ld MB\n", unit_InOutNode_size / 1024 / 1024);
    printf("unit_thread_write_size = %ld MB\n", unit_thread_write_size / 1024 / 1024);

    size_t unitCpuRam = num_ch * (unit_sacdata_size +
                                  unit_spectrum_size +
                                  unit_InOutNode_size +
                                  unit_thread_write_size +
                                  unit_thread_read_size);

    size_t h_batch = EstimateCpuBatch(unitCpuRam, argument.gpu_num);
    printf("CPU: h_batch = %ld\n", h_batch);
    // Allocate memory for GPU
    size_t wh_flag = wh_after || wh_before;
    size_t d_batch = EstimateGpuBatch(gpu_id, npts, segment_pts_1x, nstep_valid, num_ch, filter_count, wh_flag, do_runabs_mf || do_runabs);
    // print vallid batch
    printf("GPU no.%zu: nValid_batch = %ld\n", gpu_id, nValid_batch);
    h_batch = (h_batch > nValid_batch) ? nValid_batch : h_batch;
    printf("GPU no.%zu: d_batch = %ld, h_batch = %ld, num_ch = %d\n", gpu_id, d_batch, h_batch, num_ch);
    d_batch = (d_batch > h_batch) ? h_batch : d_batch;
    h_batch = d_batch;
    size_t batch = h_batch;
    batch = batch < 1 ? 1 : batch;

    printf("[GPU no.%zu]: batch = %ld, num_ch=%d\n", gpu_id, batch, num_ch);

    // Allocate CPU memory
    CpuMalloc((void **)&pInOutList, num_ch * batch * unit_InOutNode_size);
    CpuMalloc((void **)&h_sacdata, num_ch * batch * unit_sacdata_size);
    CpuMalloc((void **)&h_spectrum, num_ch * batch * unit_spectrum_size);

    // Initializing the memory for InOutNode, batch *num_ch datas will be processed at same time
    size_t sacpathSize = MAXPATH * sizeof(char);
    size_t specpathSize = MAXPATH * sizeof(char);
    size_t spechdSize = sizeof(SEGSPEC);
    size_t sachdSize = sizeof(SACHEAD);
    for (size_t i = 0; i < batch * num_ch; i++)
    {
        CpuMalloc((void **)&(pInOutList[i].sacpath), sacpathSize);
        CpuMalloc((void **)&(pInOutList[i].specpath), specpathSize);

        CpuMalloc((void **)&(pInOutList[i].sac_hd), sachdSize);
        CpuMalloc((void **)&(pInOutList[i].segspec_hd), spechdSize);

        pInOutList[i].sac_data = h_sacdata + i * npts;
        pInOutList[i].spectrum = h_spectrum + i * nstep_valid * nspec_output;

        pInOutList[i].nspec = nspec_output;
        pInOutList[i].nstep = nstep_valid;
        pInOutList[i].df = df_2x;
        pInOutList[i].dt = delta;
    }

    // ********** Allocate memory for GPU ************
    float *d_sacdata = NULL;          // segment sacdata in GPU
    float *d_sacdata_2x = NULL;       // 2x  length segment sacdata in GPU
    float *d_filtered_sacdata = NULL; // filteredsegment sacdata in GPU
    float *d_total_sacdata = NULL;    // summed sacdata of all bands in GPU

    cuComplex *d_spectrum = NULL;    // segment spectrum in GPU
    cuComplex *d_spectrum_2x = NULL; // 2x  length segment spectrum in GPU

    float *d_weight = NULL;     // weight of each segment spectrum data and sac data
    float *d_tmp = NULL;        // used in runabs
    float *d_tmp_weight = NULL; // used in runabs and whiten, store weight of single channel data

    cuComplex *d_responses = NULL; // butterworth 2th filter responses, zero-padded segment length

    double *d_sum = NULL;  // used in rtr and rdc
    double *d_isum = NULL; // used in rtr and rdc

    cufftHandle planfwd;    // forward fft
    cufftHandle planinv;    // inverse fft
    cufftHandle planfwd_2x; // forward fft, for 2x zero-padding series
    cufftHandle planinv_2x; // inverse fft, for 2x zero-padding series

    // Allocating memory in GPU Device
    AllocateGpuMemory(batch, segment_pts_1x, num_ch, do_runabs || do_runabs_mf, wh_flag,
                      &d_sacdata, &d_spectrum,
                      &d_sacdata_2x, &d_spectrum_2x,
                      &d_filtered_sacdata,
                      &d_total_sacdata,
                      &d_responses, &d_tmp,
                      &d_weight, &d_tmp_weight,
                      filter_count, &d_sum, &d_isum,
                      &planfwd, &planinv,
                      &planfwd_2x, &planinv_2x);

    // copy each filter response to GPU and store freq_low array
    float freq_lows[filter_count];
    for (int i = 0; i < filter_count; i++)
    {
        freq_lows[i] = myResp[i].freq_low;
        CUDACHECK(cudaMemcpy(d_responses + i * segment_pts_2x, myResp[i].response, segment_pts_2x * sizeof(cuComplex), cudaMemcpyHostToDevice));
    }

    // ********** MAIN PROCESSING PART ********** /
    for (size_t finish_batch = 0; finish_batch < nValid_batch; finish_batch += batch)
    {
        size_t proc_batch = (finish_batch + batch > nValid_batch) ? nValid_batch - finish_batch : batch;

        // Setup Input Output Chain
        size_t finish_cnt = finish_batch * num_ch;
        size_t proc_cnt = proc_batch * num_ch;
        for (size_t i = finish_cnt, j = 0; i < finish_cnt + proc_cnt; i++, j++)
        {
            pInOutList[j].sacpath = InPaths.paths[i];
            pInOutList[j].specpath = OutPaths.paths[i];
        }
        memset(h_sacdata, 0, proc_cnt * npts * sizeof(float));
        memset(h_spectrum, 0, proc_cnt * nstep_valid * nspec_output * sizeof(complex));

        ThreadPoolRead *read_pool = create_threadpool_read(argument.thread_num);
        ThreadPoolWrite *write_pool = create_threadpool_write(argument.thread_num);
        parallel_read_sac(read_pool, proc_cnt, pInOutList, argument.thread_num); // Parallel read SAC data

        dim3 dimgrd_1x, dimblk_1x, dimgrd_2x, dimblk_2x;
        DimCompute(&dimgrd_1x, &dimblk_1x, segment_pts_1x, proc_cnt);
        DimCompute(&dimgrd_2x, &dimblk_2x, segment_pts_2x, proc_cnt);
        for (int stepidx = 0, done_step = 0; stepidx < nstep; stepidx++)
        {
            int skip_this_step = 0;
            for (int flag_i = 0; flag_i < argument.skip_step_count; flag_i++)
            {
                if (argument.skip_steps[flag_i] == stepidx)
                {
                    skip_this_step = 1;
                    break;
                }
            }

            if (skip_this_step)
            {
                continue;
            }

            // clean d_sacdata_* and d_spectrum_*
            CUDACHECK(cudaMemset(d_sacdata, 0, proc_cnt * segment_pts_1x * sizeof(float)));
            CUDACHECK(cudaMemset(d_spectrum, 0, proc_cnt * segment_pts_1x * sizeof(cuComplex)));
            CUDACHECK(cudaMemset(d_sacdata_2x, 0, proc_cnt * segment_pts_2x * sizeof(float)));
            CUDACHECK(cudaMemset(d_spectrum_2x, 0, proc_cnt * segment_pts_2x * sizeof(cuComplex)));

            // copy sacdata to GPU
            CUDACHECK(cudaMemcpy2D(d_sacdata, segment_pts_1x * sizeof(float),
                                   h_sacdata + stepidx * shift_length_pts, npts * sizeof(float),
                                   segment_pts_1x * sizeof(float), proc_cnt, cudaMemcpyHostToDevice));
            preprocess(d_sacdata, d_sum, d_isum, segment_pts_1x, proc_cnt, freq_low, delta); // isnan,rtr,reman,taper

            // double zero-padding before filtering, add 20241216 by wangjx
            CUDACHECK(cudaMemcpy2D(d_sacdata_2x, segment_pts_2x * sizeof(float), d_sacdata, segment_pts_1x * sizeof(float), segment_pts_1x * sizeof(float), proc_cnt, cudaMemcpyDeviceToDevice));
            CUFFTCHECK(cufftExecR2C(planfwd_2x, (cufftReal *)d_sacdata_2x, (cufftComplex *)d_spectrum_2x));
            FwdNormalize2DKernel<<<dimgrd_2x, dimblk_2x>>>(d_spectrum_2x, segment_pts_2x, segment_pts_2x, proc_cnt, delta);
            cisnan2DKernel<<<dimgrd_2x, dimblk_2x>>>(d_spectrum_2x, segment_pts_2x, segment_pts_2x, proc_cnt);
            dim3 c_fdimgrd, c_fdimblk;                   // Filtering add 20240711 by wangjx
            size_t fwidth_2x = 0.5 * segment_pts_2x + 1; // set to 0.5 * segment_pts_2x + 1, because of the zero-padding
            DimCompute(&c_fdimgrd, &c_fdimblk, fwidth_2x, proc_cnt);
            filterKernel<<<c_fdimgrd, c_fdimblk>>>(d_spectrum_2x, d_responses, segment_pts_2x, fwidth_2x, proc_cnt);
            CUFFTCHECK(cufftExecC2R(planinv_2x, (cufftComplex *)d_spectrum_2x, (cufftReal *)d_sacdata_2x));
            InvNormalize2DKernel<<<dimgrd_2x, dimblk_2x>>>(d_sacdata_2x, segment_pts_2x, segment_pts_2x, proc_cnt, delta);

            // cut the data to nseg length
            CUDACHECK(cudaMemcpy2D(d_sacdata, segment_pts_1x * sizeof(float), d_sacdata_2x, segment_pts_2x * sizeof(float), segment_pts_1x * sizeof(float), proc_cnt, cudaMemcpyDeviceToDevice));

            if (wh_before)
            {
                CUFFTCHECK(cufftExecR2C(planfwd, (cufftReal *)d_sacdata, (cufftComplex *)d_spectrum));
                FwdNormalize2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_spectrum, segment_pts_1x, segment_pts_1x, proc_cnt, delta);
                freqWhiten(d_spectrum, d_weight, d_tmp_weight, d_tmp, num_ch, segment_pts_1x, proc_batch, delta, f_idx1, f_idx2, f_idx3, f_idx4);
                cisnan2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_spectrum, segment_pts_1x, segment_pts_1x, proc_cnt);
                CUFFTCHECK(cufftExecC2R(planinv, (cufftComplex *)d_spectrum, (cufftReal *)d_sacdata));
                InvNormalize2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_sacdata, segment_pts_1x, segment_pts_1x, proc_cnt, delta);
            }

            //   Time Domain Normalization
            if (do_runabs_mf)
            {
                runabs_mf(d_sacdata, d_filtered_sacdata, d_total_sacdata,
                          d_sacdata_2x, d_spectrum_2x,
                          d_responses, d_tmp,
                          d_weight, d_tmp_weight,
                          &planinv, freq_lows,
                          filter_count, delta, proc_batch, num_ch, MAXVAL, segment_pts_1x, segment_pts_2x, &planinv_2x, &planfwd_2x);
                CUDACHECK(cudaMemcpy2D(d_sacdata, segment_pts_1x * sizeof(float),
                                       d_total_sacdata, segment_pts_1x * sizeof(float),
                                       segment_pts_1x * sizeof(float), proc_cnt, cudaMemcpyDeviceToDevice));
            }

            if (do_onebit)
            {
                onebit2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_sacdata, segment_pts_1x, segment_pts_1x, proc_cnt);
            }

            if (do_runabs)
            {
                float freq_lows_limit = freq_low * 0.667;
                runabs(d_sacdata, d_tmp, d_weight, d_tmp_weight, freq_lows_limit, delta, proc_batch, num_ch, segment_pts_1x, MAXVAL);
            }

            if (wh_after)
            {
                CUFFTCHECK(cufftExecR2C(planfwd, (cufftReal *)d_sacdata, (cufftComplex *)d_spectrum));
                FwdNormalize2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_spectrum, segment_pts_1x, segment_pts_1x, proc_cnt, delta);
                freqWhiten(d_spectrum, d_weight, d_tmp_weight, d_tmp, num_ch, segment_pts_1x, proc_batch, delta, f_idx1, f_idx2, f_idx3, f_idx4);
                cisnan2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_spectrum, segment_pts_1x, segment_pts_1x, proc_cnt);
                CUFFTCHECK(cufftExecC2R(planinv, (cufftComplex *)d_spectrum, (cufftReal *)d_sacdata));
                InvNormalize2DKernel<<<dimgrd_1x, dimblk_1x>>>(d_sacdata, segment_pts_1x, segment_pts_1x, proc_cnt, delta);
            }

            // Zero-Padding
            CUDACHECK(cudaMemcpy2D(d_sacdata_2x, segment_pts_2x * sizeof(float), d_sacdata, segment_pts_1x * sizeof(float), segment_pts_1x * sizeof(float), proc_cnt, cudaMemcpyDeviceToDevice));
            CUFFTCHECK(cufftExecR2C(planfwd_2x, (cufftReal *)d_sacdata_2x, (cufftComplex *)d_spectrum_2x));
            FwdNormalize2DKernel<<<dimgrd_2x, dimblk_2x>>>(d_spectrum_2x, segment_pts_2x, segment_pts_2x, proc_cnt, delta);
            cisnan2DKernel<<<dimgrd_2x, dimblk_2x>>>(d_spectrum_2x, segment_pts_2x, segment_pts_2x, proc_cnt);

            // Copy data from d_spectrum back to h_spectrum
            CUDACHECK(cudaMemcpy2D(h_spectrum + done_step * nspec_output, nstep_valid * nspec_output * sizeof(cuComplex),
                                   d_spectrum_2x, segment_pts_2x * sizeof(cuComplex),
                                   nspec_output * sizeof(complex), proc_cnt, cudaMemcpyDeviceToHost));

            done_step++;
        }

        // End loop of each segment, Writing the output spectrum
        parallel_write_spec(write_pool, proc_cnt, pInOutList, argument.thread_num);
        destroy_threadpool_write(write_pool);
        destroy_threadpool_read(read_pool);
    }
    // Free memory
    cufftDestroy(planfwd);
    cufftDestroy(planinv);
    cufftDestroy(planfwd_2x);
    GpuFree((void **)&d_sacdata);
    GpuFree((void **)&d_spectrum);
    GpuFree((void **)&d_sacdata_2x);
    GpuFree((void **)&d_spectrum_2x);
    GpuFree((void **)&d_filtered_sacdata);
    GpuFree((void **)&d_total_sacdata);
    GpuFree((void **)&d_responses);
    GpuFree((void **)&d_weight);
    GpuFree((void **)&d_tmp);
    GpuFree((void **)&d_tmp_weight);
    GpuFree((void **)&d_sum);
    GpuFree((void **)&d_isum);
    CpuFree((void **)&h_sacdata);
    CpuFree((void **)&h_spectrum);
    CpuFree((void **)&pInOutList);
}