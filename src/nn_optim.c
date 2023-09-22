#include "nn_optim.h"

#include <assert.h>

nn_optim_t *nn_optim_construct(nn_optim_t *optimizer, const nn_model_t *model, FLT_TYP optim_rate)
{
    assert(optimizer);
    assert(model);
    assert(optim_rate > 0);
    optimizer->construct(optimizer, model, optim_rate);
    return optimizer;
}

void nn_optim_destruct(nn_optim_t *optimizer)
{
    assert(optimizer);
    optimizer->destruct(optimizer);
}

nn_optim_t nn_optim_SGD;
