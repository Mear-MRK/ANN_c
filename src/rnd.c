#include "rnd.h"

#include <assert.h>

uint64_t pcg_uint64(void)
{
    uint64_t r = pcg_uint32();
    r = (r << 32) ^ pcg_uint32();
    return r;
}

FLT_TYP uniform_flt_rnd(const void *param)
{
    if (!param)
        return UNIF_FLT_RND_GEN();

    const flt_rnd_param *p = (const flt_rnd_param *)param;
    return p->amp * (2 * UNIF_FLT_RND_GEN() - 1) + p->mean;
}

IND_TYP int_rnd(IND_TYP a, IND_TYP b)
{
    assert(a != b);
    IND_TYP dif = b - a;
    IND_TYP drc;
    if (dif >= 0)
        drc = 1;
    else
    {
        drc = -1;
        dif = -dif;
    }
    if ((uint64_t)dif <= PCG_UINT32_MAX)
        return a + (IND_TYP)(pcg_uint32() % (uint32_t)dif) * drc;
    else
        return a + (IND_TYP)(pcg_uint64() % (uint64_t)dif) * drc;
}
