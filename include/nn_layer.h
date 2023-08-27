#ifndef NN_LAYER_T_H_INCLUDED
#define NN_LAYER_T_H_INCLUDED 1

#include "lin_alg.h"
#include "nn_activ.h"

typedef struct nn_layer_struct
{
    IND_TYP out_sz;
    nn_activ_t activ;
    FLT_TYP dropout;
} nn_layer_t;

extern const nn_layer_t nn_layer_NULL;

nn_layer_t *nn_layer_init(
    nn_layer_t *layer,
    IND_TYP output_size,
    nn_activ_t activation,
    FLT_TYP dropout_ratio);

#endif /* NN_LAYER_T_H_INCLUDED */