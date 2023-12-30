#include "nn_optim.h"

#include <assert.h>
#include <string.h>

#include "nn_model.h"

nn_optim *nn_optim_construct(nn_optim *optimizer, const nn_optim_class *optim_class, const nn_model *model)
{
    assert(optimizer);
    assert(model);
    assert(optim_class);
    optimizer->class = *optim_class;
    if (optimizer->class.construct)
        return optimizer->class.construct(optimizer, model);
    return optimizer;
}

nn_optim *nn_optim_set_params(nn_optim *optimizer, const void *params)
{
    assert(optimizer);
    if (optimizer->class.set_params)
        return optimizer->class.set_params(optimizer, params);
    return optimizer;
}

void nn_optim_destruct(nn_optim *optimizer)
{
    assert(optimizer);
    if (optimizer->class.destruct)
        optimizer->class.destruct(optimizer);
    *optimizer = nn_optim_NULL;
}

nn_model *nn_optim_update_model(nn_optim *optimizer, nn_model *model)
{
    assert(optimizer);
    assert(model);
    return optimizer->class.update_model(optimizer, model);
}
