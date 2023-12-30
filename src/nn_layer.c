#include "nn_layer.h"

#include <assert.h>
#include <string.h>
#include <stdio.h>

nn_layer *nn_layer_init(
    nn_layer *layer,
    IND_TYP output_size,
    nn_activ activation,
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

bool nn_layer_is_null(const nn_layer *layer)
{
    assert(layer);
    return memcmp(layer, &nn_layer_NULL, sizeof(nn_layer)) == 0;
}

char *nn_layer_to_str(const nn_layer *layer, char *string)
{
    assert(layer);
    assert(string);
    string[0] = 0;
    char buff[32];
    buff[0] = 0;
    sprintf(string, "nn_layer: ouput size: %ld, dropout %g, activation: %s",
            layer->out_sz, layer->dropout, nn_activ_to_str(&layer->activ, buff));
    return string;
}

size_t nn_layer_serial_size(const nn_layer *layer)
{
    assert(layer);
    return sizeof(size_t) + sizeof(layer->out_sz) + sizeof(enum nn_activ_enum) + sizeof(layer->dropout);
}

uint8_t *nn_layer_serialize(const nn_layer *layer, uint8_t *byte_arr)
{
    assert(layer);
    assert(byte_arr);
    size_t sz = nn_layer_serial_size(layer);
    memcpy(byte_arr, &sz, sizeof(size_t));
    byte_arr += sizeof(size_t);
    size_t sz_o = sizeof(layer->out_sz);
    size_t sz_a = sizeof(enum nn_activ_enum);
    size_t sz_d = sizeof(layer->dropout);
    memcpy(byte_arr, &layer->out_sz, sz_o);
    byte_arr += sz_o;
    enum nn_activ_enum a = nn_activ_to_enum(&layer->activ);
    memcpy(byte_arr, &a, sz_a);
    byte_arr += sz_a;
    memcpy(byte_arr, &layer->dropout, sz_d);
    byte_arr += sz_d;
    return byte_arr;
}

const uint8_t *nn_layer_deserialize(nn_layer *layer, const uint8_t *byte_arr)
{
    assert(layer);
    assert(byte_arr);
    byte_arr += sizeof(size_t);
    size_t sz_o = sizeof(layer->out_sz);
    size_t sz_a = sizeof(enum nn_activ_enum);
    size_t sz_d = sizeof(layer->dropout);
    memcpy(&layer->out_sz, byte_arr, sz_o);
    byte_arr += sz_o;
    enum nn_activ_enum a;
    memcpy(&a, byte_arr, sz_a);
    byte_arr += sz_a;
    layer->activ = nn_activ_from_enum(a);
    memcpy(&layer->dropout, byte_arr, sz_d);
    byte_arr += sz_d;
    return byte_arr;
}
