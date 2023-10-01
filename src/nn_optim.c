#include "nn_optim.h"

#include <assert.h>
#include <string.h>

#include "nn_model.h"

nn_optim_t *nn_optim_construct(nn_optim_t *optimizer, const nn_optim_class *optim_class, const nn_model_t *model)
{
    assert(optimizer);
    assert(model);
    assert(optim_class);
    optimizer->class = *optim_class;
    return optimizer->class.construct(optimizer, model);
}

nn_optim_t *nn_optim_set_params(nn_optim_t *optimizer, const void *params)
{
    if (optimizer->class.set_params)
        return optimizer->class.set_params(optimizer, params);
    return optimizer;
}

void nn_optim_destruct(nn_optim_t *optimizer)
{
    assert(optimizer);
    optimizer->class.destruct(optimizer);
    *optimizer = nn_optim_NULL;
}

nn_model_t *nn_optim_update_model(nn_optim_t *optimizer, nn_model_t *model)
{
    assert(optimizer);
    assert(model);
    return optimizer->class.update_model(optimizer, model);
}
