#ifndef NN_OPTIM_H_INCLUDED
#define NN_OPTIM_H_INCLUDED 1

#include "nn_config.h"
// #include "nn_model.h"

struct nn_model;
typedef struct nn_model nn_model;

struct nn_optim_class;
typedef struct nn_optim_class nn_optim_class;

struct nn_optim;
typedef struct nn_optim nn_optim;

typedef nn_model *(*nn_optim_update_model_func)(nn_optim *optimizer, nn_model *model);
typedef nn_optim *(*nn_optim_construct_func)(nn_optim *optimizer, const nn_model *model);
typedef nn_optim *(*nn_optim_set_params_func)(nn_optim *optimizer, const void *params);
typedef void (*nn_optim_destruct_func)(nn_optim *optimizer);

struct nn_optim_class
{
    nn_optim_update_model_func update_model;
    nn_optim_construct_func construct;
    nn_optim_set_params_func set_params;
    nn_optim_destruct_func destruct;
};

#define nn_optim_class_NULL \
((const nn_optim_class){.update_model = NULL, .construct = NULL, .set_params = NULL, .destruct = NULL})

struct nn_optim
{
    nn_optim_class class;
    void *params;
    void *intern;
};

#define nn_optim_NULL ((const nn_optim){.class = nn_optim_class_NULL, .params = NULL, .intern = NULL})

nn_optim *nn_optim_construct(nn_optim *optimizer, const nn_optim_class *optim_class, const nn_model *model);
nn_optim *nn_optim_set_params(nn_optim *optimizer, const void *params);
void nn_optim_destruct(nn_optim *optimizer);
nn_model *nn_optim_update_model(nn_optim *optimizer, nn_model *model);

#endif /* NN_OPTIM_H_INCLUDED */