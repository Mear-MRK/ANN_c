#include "nn_optim_cls_SGD.h"

#include <stdlib.h>
#include <assert.h>

#include "nn_model.h"

static nn_optim *nn_optim_cls_SGD_construct(nn_optim *optimizer, const nn_model *model)
{
    assert(optimizer);
    optimizer->intern = NULL;
    optimizer->params = calloc(1, sizeof(nn_optim_cls_SGD_params));
    assert(optimizer->params);
    ((nn_optim_cls_SGD_params *)optimizer->params)->learning_rate = 0.0001f;
    return optimizer;
}

static void nn_optim_cls_SGD_destruct(nn_optim *optimizer)
{
    assert(optimizer);
    if (optimizer->params)
        free(optimizer->params);
    optimizer->params = NULL;
}

static nn_optim *nn_optim_cls_SGD_set_params(nn_optim *optimizer, const void *inp_params)
{
    assert(optimizer);
    nn_optim_cls_SGD_params *params = (nn_optim_cls_SGD_params *)optimizer->params;
    nn_optim_cls_SGD_params *inp_p = (nn_optim_cls_SGD_params *)inp_params;
    if (inp_p && inp_p->learning_rate > 0)
    {
        params->learning_rate = inp_p->learning_rate;
    }
    return optimizer;
}

static nn_model *nn_optim_cls_SGD_update_model(nn_optim *optimizer, nn_model *model)
{
    assert(optimizer);
    assert(model);
    nn_optim_cls_SGD_params *params = (nn_optim_cls_SGD_params *)optimizer->params;
    FLT_TYP alpha = -params->learning_rate;
    for (int l = 0; l < model->nbr_layers; l++)
    {
        mat_update(model->weight + l, alpha, model->intern.d_w + l);
        vec_update(model->bias + l, alpha, model->intern.d_b + l);
    }
    return model;
}

const nn_optim_class nn_optim_cls_SGD = {.construct = nn_optim_cls_SGD_construct,
                                         .destruct = nn_optim_cls_SGD_destruct,
                                         .set_params = nn_optim_cls_SGD_set_params,
                                         .update_model = nn_optim_cls_SGD_update_model};
