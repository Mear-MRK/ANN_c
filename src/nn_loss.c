#include "nn_loss.h"

#include <assert.h>
#include <string.h>

nn_loss_t *nn_loss_init(nn_loss_t *err, const nn_loss_func err_func, const nn_deriv_loss_func deriv)
{
    assert(err);
    assert(err_func);
    assert(deriv);
    err->func = err_func;
    err->deriv = deriv;
    return err;
}

enum nn_loss_enum nn_loss_to_enum(const nn_loss_t *err)
{
    assert(err);
    if (memcmp(err, &nn_loss_MSE, sizeof(nn_loss_t)) == 0)
        return LOSS_MSE;
    return LOSS_NON;
}

nn_loss_t nn_loss_from_enum(const enum nn_loss_enum e)
{
    switch (e)
    {
    case LOSS_MSE:
        return nn_loss_MSE;
    case LOSS_CCE:
        return nn_loss_CrossEnt;
    }
    return nn_loss_NULL;
}

static inline FLT_TYP mse_f(const vec_t *trg, const vec_t *out, vec_t *buff)
{
    assert(vec_is_valid(buff));
    return vec_norm_2(vec_sub(buff, out, trg));
}
static inline vec_t *mse_drv(vec_t *res, const vec_t *trg, const vec_t *out)
{
    return vec_scale(vec_sub(res, out, trg), 2);
}

const nn_loss_t nn_loss_MSE = {.func = mse_f, .deriv = mse_drv};

static inline FLT_TYP cce_f(const vec_t *trg, const vec_t *out, vec_t *buff)
{
    assert(buff);
    vec_log2(buff, vec_softmax(buff, out));
    return -vec_dot(trg, buff);
}
static inline vec_t *cce_drv(vec_t *res, const vec_t *trg, const vec_t *out)
{
    vec_softmax(res, out);
    return vec_sub(res, res, trg);
}

const nn_loss_t nn_loss_CrossEnt = {.func = cce_f, .deriv = cce_drv};