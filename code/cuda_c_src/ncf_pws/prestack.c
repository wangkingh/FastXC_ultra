#include "prestack.h"
#include "sac.h" /* swab4 / BYTE_SWAP */
#include <stdlib.h>
#include <string.h>

#ifdef BYTE_SWAP
#define SWAP4(p, n) swab4((char *)(p), (int)((n) * sizeof(float)))
#else
#define SWAP4(p, n) ((void)0)
#endif

/* A. 均匀分组 ------------------------------------------------ */
int group_offsets_uniform(OffItem *all, unsigned num_segments,
                          unsigned grp_sz,
                          OffGroup **pGroups, unsigned *pNg)
{
    if (grp_sz < 1)
        grp_sz = 1;
    unsigned ng = (num_segments + grp_sz - 1) / grp_sz;

    OffGroup *G = malloc(ng * sizeof *G);
    if (!G)
        return -1;

    for (unsigned g = 0; g < ng; ++g)
    {
        unsigned first = g * grp_sz;
        unsigned sz = (first + grp_sz <= num_segments) ? grp_sz
                                                       : (num_segments - first);
        G[g].arr = all + first;
        G[g].size = sz;
    }
    *pGroups = G;
    *pNg = ng;
    return 0;
}

/* helper */
void free_groups(OffGroup *g) { free(g); }

/* B. 叠加单组 ------------------------------------------------ */
int stack_group_average(FILE *fp, OffGroup *grp,
                        unsigned ns, float *out, int do_swap)
{
    float *buf = malloc(ns * sizeof(float));
    if (!buf)
        return -1;
    memset(out, 0, ns * sizeof(float));

    for (unsigned k = 0; k < grp->size; ++k)
    {
        if (fseeko(fp, grp->arr[k].data_off, SEEK_SET) != 0)
        {
            free(buf);
            return -1;
        }
        if (fread(buf, sizeof(float), ns, fp) != ns)
        {
            free(buf);
            return -1;
        }
        if (do_swap)
            SWAP4(buf, ns);

        for (unsigned j = 0; j < ns; ++j)
            out[j] += buf[j];
    }
    float inv = 1.0f / (float)grp->size;
    for (unsigned j = 0; j < ns; ++j)
        out[j] *= inv;

    free(buf);
    return 0;
}

/* C. 填充整体矩阵 ------------------------------------------- */
int fill_prestack_matrix(FILE *fp,
                         OffGroup *groups,
                         unsigned ngroups,
                         unsigned nsamples,
                         int do_swap,
                         float *matrix_out)
{
    for (unsigned g = 0; g < ngroups; ++g)
    {
        if (stack_group_average(fp, &groups[g], nsamples,
                                matrix_out + (size_t)g * nsamples,
                                do_swap) != 0)
        {
            return -1;
        }
    }
    return 0;
}
