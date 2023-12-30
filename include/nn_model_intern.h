#pragma once

#include <stddef.h>

#include "nn_config.h"
#include "lin_alg.h"
#include "nn_layer.h"

typedef struct nn_model_intern
{
    int nbr_layers;
    mat *d_w;
    vec *d_b;
    vec *s;
    vec *a;
    vec *a_mask;
    vec a_inp;
} nn_model_intern;

#define nn_model_intern_NULL ((const nn_model_intern){.nbr_layers = 0, .d_w = NULL, .d_b = NULL, .s = NULL, .a = NULL, .a_mask = NULL})

nn_model_intern *nn_model_intern_construct(nn_model_intern *intern, int layer_capacity, IND_TYP inp_size);

void nn_model_intern_destruct(nn_model_intern *intern);

nn_model_intern *nn_model_intern_add(nn_model_intern *intern, const nn_layer *layer, IND_TYP input_size);
nn_model_intern *nn_model_intern_remove(nn_model_intern *intern, int layer_index);

void nn_model_reset_gradients(nn_model_intern *intern);