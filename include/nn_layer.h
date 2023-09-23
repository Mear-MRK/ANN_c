#ifndef NN_LAYER_T_H_INCLUDED
#define NN_LAYER_T_H_INCLUDED 1

#include "lin_alg.h"
#include "nn_activ.h"
#include "byte_arr.h"

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

char *nn_layer_to_str(const nn_layer_t* layer, char *string);

size_t nn_layer_serial_size(const nn_layer_t* layer);
// returna a pointer to the byte after the last byte written
uint8_t *nn_layer_serialize(const nn_layer_t* layer, uint8_t *byte_arr);
// returns a pointer to the byte after the last byte read
const uint8_t *nn_layer_deserialize(nn_layer_t *layer, const uint8_t *byte_arr);

#endif /* NN_LAYER_T_H_INCLUDED */