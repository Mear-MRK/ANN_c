#include "nn_layer.h"

#include <assert.h>
#include <string.h>

const nn_layer_t nn_layer_NULL =
    {.out_sz = 0, .activ = nn_activ_NULL, .dropout = 0};



nn_layer_t *nn_layer_init(
    nn_layer_t *layer, 
    size_t output_size, 
    nn_activ_t activation,
    FLT_TYP dropout_ratio)
{
    assert(layer);
    assert(output_size);
    assert(dropout_ratio >= 0 && dropout_ratio < 1);

    layer->out_sz = output_size;
    layer->activ = activation;
    layer->dropout = dropout_ratio;

    return layer;
}

