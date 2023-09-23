#include "nn_layer.h"

#include <assert.h>
#include <string.h>
#include <stdio.h>

const nn_layer_t nn_layer_NULL =
    {.out_sz = 0, .activ = nn_activ_NULL, .dropout = 0};

nn_layer_t *nn_layer_init(
    nn_layer_t *layer,
    IND_TYP output_size,
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

char *nn_layer_to_str(const nn_layer_t *layer, char *string)
{
    assert(layer);
    assert(string);
    string[0] = 0;
    char buff[32];
    buff[0] = 0;
    sprintf(string, "nn_layer: ouput size: %d, dropout %g, activation: %s",
            layer->out_sz, layer->dropout, nn_activ_to_str(&layer->activ, buff));
    return string;
}

size_t nn_layer_serial_size(const nn_layer_t *layer)
{
    assert(layer);
    return sizeof(layer->out_sz) + sizeof(enum nn_activ) + sizeof(layer->dropout);
}

uint8_t *nn_layer_serialize(const nn_layer_t *layer, uint8_t *byte_arr)
{
    assert(layer);
    assert(byte_arr);
    size_t sz_o = sizeof(layer->out_sz);
    size_t sz_a = sizeof(enum nn_activ);
    size_t sz_d = sizeof(layer->dropout);
    memcpy(byte_arr, &layer->out_sz, sz_o);
    byte_arr += sz_o;
    enum nn_activ a = nn_activ_to_enum(&layer->activ);
    memcpy(byte_arr, &a, sz_a);
    byte_arr += sz_a;
    memcpy(byte_arr, &layer->dropout, sz_d);
    byte_arr += sz_d;
    return byte_arr;
}

const uint8_t *nn_layer_deserialize(nn_layer_t *layer, const uint8_t *byte_arr)
{
    assert(layer);
    assert(byte_arr);
    size_t sz_o = sizeof(layer->out_sz);
    size_t sz_a = sizeof(enum nn_activ);
    size_t sz_d = sizeof(layer->dropout);
    memcpy(&layer->out_sz, byte_arr, sz_o);
    byte_arr += sz_o;
    enum nn_activ a;
    memcpy(&a, byte_arr, sz_a);
    byte_arr += sz_a;
    layer->activ = nn_activ_from_enum(a);
    memcpy(&layer->dropout, byte_arr, sz_d);
    byte_arr += sz_d;
    return byte_arr;
}
