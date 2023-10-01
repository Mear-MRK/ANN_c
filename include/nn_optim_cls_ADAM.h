#pragma once

#include "nn_optim.h"

#include "lin_alg.h"

extern const nn_optim_class nn_optim_cls_ADAM;

typedef struct nn_optim_cls_ADAM_params_struct
{
    FLT_TYP alpha;
    FLT_TYP beta1, beta2;
    FLT_TYP eps;
    unsigned t0;
} nn_optim_cls_ADAM_params_t;

#define nn_optim_cls_ADAM_params_DEFAULT ((const nn_optim_cls_ADAM_params_t)\
{.alpha = 0.001, .beta1 = 0.9, .beta2 = 0.999, .eps = 1.0E-8, .t0 = 0})

void nn_optim_cls_ADAM_params_clear(nn_optim_cls_ADAM_params_t *params);
