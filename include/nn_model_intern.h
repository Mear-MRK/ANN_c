#pragma once

#include <stddef.h>

#include "lin_alg.h"
#include "nn_layer.h"
#include "nn_conf.h"

typedef struct nn_model_intern
{
    int nbr_layers;
    mat_t *d_w;
    vec_t *d_b;
    vec_t *s;
    vec_t *a;
    vec_t *a_mask;
    vec_t a_inp;
} nn_model_intern_t;

#define nn_model_intern_NULL ((const nn_model_intern_t){.nbr_layers = 0, .d_w = NULL, .d_b = NULL, .s = NULL, .a = NULL, .a_mask = NULL})

nn_model_intern_t *nn_model_intern_construct(nn_model_intern_t *intern, int layer_capacity, IND_TYP inp_size);

void nn_model_intern_destruct(nn_model_intern_t *intern);

nn_model_intern_t *nn_model_intern_add(nn_model_intern_t *intern, const nn_layer_t *layer, IND_TYP input_size);
nn_model_intern_t *nn_model_intern_remove(nn_model_intern_t *intern, int layer_index);

void nn_model_reset_gradients(nn_model_intern_t *intern);