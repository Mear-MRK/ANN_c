#ifndef NN_LAYER_T_H_INCLUDED
#define NN_LAYER_T_H_INCLUDED 1

#include "lin_alg.h"

typedef struct nn_layer_struct
{
    size_t inp_sz;
    size_t out_sz;
    vec_t *(*act)(vec_t *result, const vec_t *v);
    vec_t *(*drv_act)(vec_t *result, const vec_t *v, const vec_t *act_v);
    FLT_TYPE dropout;
} nn_layer_t;

extern const nn_layer_t nn_layer_NULL;

nn_layer_t *nn_layer_init(
    nn_layer_t *layer,
    size_t input_size, size_t output_size,
    vec_t *(*activation)(vec_t *result, const vec_t *v),
    vec_t *(*derivitive_of_activation)(vec_t *result, const vec_t *v, const vec_t *activation_result),
    FLT_TYPE dropout_ratio);

#endif /* NN_LAYER_T_H_INCLUDED */