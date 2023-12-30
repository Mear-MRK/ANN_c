#pragma once

#include "nn_config.h"
#include "pcg.h"

#ifdef FLD_FLT64
#define UNIF_FLT_RND_GEN pcg_dbl
#else
#define UNIF_FLT_RND_GEN pcg_flt
#endif

#ifdef IND_INT32
#define UINT_RND_GEN pcg_uint32
#else
#define UINT_RND_GEN pcg_uint64
#endif

uint64_t pcg_uint64(void);

typedef struct flt_rnd_param
{
    FLT_TYP amp;
    FLT_TYP mean;
} flt_rnd_param;


FLT_TYP uniform_flt_rnd(const void *param);
IND_TYP int_rnd(IND_TYP a, IND_TYP b);
