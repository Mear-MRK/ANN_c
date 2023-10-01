#pragma once

#include "nn_optim.h"

#include "lin_alg.h"

extern const nn_optim_class nn_optim_cls_SGD;

typedef struct nn_optim_cls_SGD_params_struct
{
    FLT_TYP learning_rate;
} nn_optim_cls_SGD_params_t;