#include "nn_err.h"

#include <assert.h>
#include <string.h>

nn_err_t *nn_err_init(nn_err_t *err, const nn_err_func err_func, const nn_deriv_err_func deriv)
{
    assert(err);
    assert(err_func);
    assert(deriv);
    err->func = err_func;
    err->deriv = deriv;
    return err;
}

enum nn_err nn_err_to_enum(const nn_err_t *err)
{
    assert(err);
    if (memcmp(err, &nn_err_MSE, sizeof(nn_err_t)) == 0)
        return ERR_MSE;
    return ERR_NON;
}

nn_err_t nn_err_from_enum(const enum nn_err e)
{
    switch (e)
    {
    case ERR_MSE:
        return nn_err_MSE;
    }
    return nn_err_NULL;
}

static inline FLT_TYP mse_f(const vec_t *trg, const vec_t *out, vec_t *buff)
{
    assert(buff);
    return vec_norm_2(vec_sub(buff, out, trg));
}
static inline vec_t *mse_drv(vec_t *res, const vec_t *trg, const vec_t *out)
{
    return vec_scale(vec_sub(res, out, trg), 2);
}

const nn_err_t nn_err_MSE = {.func = mse_f, .deriv = mse_drv};