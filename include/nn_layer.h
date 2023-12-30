#pragma once

#include <stdbool.h>

#include "nn_config.h"
#include "lin_alg.h"
#include "nn_activ.h"

typedef struct nn_layer
{
    IND_TYP out_sz;
    nn_activ activ;
    FLT_TYP dropout;
} nn_layer;

#define nn_layer_NULL ((const nn_layer){.out_sz = 0, .activ = nn_activ_NULL, .dropout = 0})

nn_layer *nn_layer_init(
    nn_layer *layer,
    IND_TYP output_size,
    nn_activ activation,
    FLT_TYP dropout_ratio);

bool nn_layer_is_null(const nn_layer *layer);

char *nn_layer_to_str(const nn_layer *layer, char *string);

size_t nn_layer_serial_size(const nn_layer *layer);
// returna a pointer to the byte after the last byte written
uint8_t *nn_layer_serialize(const nn_layer *layer, uint8_t *byte_arr);
// returns a pointer to the byte after the last byte read
const uint8_t *nn_layer_deserialize(nn_layer *layer, const uint8_t *byte_arr);
