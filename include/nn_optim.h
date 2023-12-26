#ifndef NN_OPTIM_H_INCLUDED
#define NN_OPTIM_H_INCLUDED 1

// #include "nn_model.h"
#include "nn_conf.h"

struct nn_model;
typedef struct nn_model nn_model_t;

struct nn_optim_class;
typedef struct nn_optim_class nn_optim_class;

struct nn_optim;
typedef struct nn_optim nn_optim_t;

typedef nn_model_t *(*nn_optim_update_model_func)(nn_optim_t *optimizer, nn_model_t *model);
typedef nn_optim_t *(*nn_optim_construct_func)(nn_optim_t *optimizer, const nn_model_t *model);
typedef nn_optim_t *(*nn_optim_set_params_func)(nn_optim_t *optimizer, const void *params);
typedef void (*nn_optim_destruct_func)(nn_optim_t *optimizer);

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

#define nn_optim_NULL ((const nn_optim_t){.class = nn_optim_class_NULL, .params = NULL, .intern = NULL})

nn_optim_t *nn_optim_construct(nn_optim_t *optimizer, const nn_optim_class *optim_class, const nn_model_t *model);
nn_optim_t *nn_optim_set_params(nn_optim_t *optimizer, const void *params);
void nn_optim_destruct(nn_optim_t *optimizer);
nn_model_t *nn_optim_update_model(nn_optim_t *optimizer, nn_model_t *model);

#endif /* NN_OPTIM_H_INCLUDED */