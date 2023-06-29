#include "nn_layer_t.h"

#include <assert.h>
#include <string.h>

const nn_layer_t nn_layer_NULL =
    {.inp_sz = 0, .out_sz = 0, .act = NULL, .drv_act = NULL, .dropout = 0};



nn_layer_t *nn_layer_init(
    nn_layer_t *layer, 
    size_t input_size, size_t output_size, 
    vec_t *(*act)(vec_t *result, const vec_t *v), 
    vec_t *(*drv_act)(vec_t *result, const vec_t *v, const vec_t *act_res), 
    FLT_TYPE dropout_ratio)
{
    assert(layer);
    assert(input_size);
    assert(output_size);
    assert(act);
    assert(drv_act);
    assert(dropout_ratio >= 0 && dropout_ratio < 1);

    layer->inp_sz = input_size;
    layer->out_sz = output_size;
    layer->act = act;
    layer->drv_act = drv_act;
    layer->dropout = dropout_ratio;

    return layer;
}

