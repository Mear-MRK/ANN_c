#pragma once

#include <stdbool.h>

#include "nn_config.h"
#include "nn_layer.h"
#include "nn_model_intern.h"
#include "data_points.h"


typedef struct nn_model
{
    IND_TYP input_size;
    IND_TYP ouput_size;
    int layer_capacity;
    int nbr_layers;
    IND_TYP max_width;
    nn_layer *layer;
    mat *weight;
    vec *bias;
    nn_model_intern intern;
} nn_model;

extern const nn_model nn_model_NULL;

#include "nn_optim.h"
#include "nn_loss.h"

bool nn_model_is_null(const nn_model *model);

nn_model *nn_model_construct(nn_model *model, int layer_capacity, IND_TYP input_size);

void nn_model_destruct(nn_model *model);

nn_model *nn_model_init_uniform_rnd(nn_model *model, FLT_TYP amp, FLT_TYP mean);
// nn_model *nn_model_init_copy(nn_model *model, const nn_model *src_model);

// append layer
nn_model *nn_model_append(nn_model *model, const nn_layer *layer);
// remove layer
nn_model *nn_model_remove(nn_model *model, int layer_index);

vec *nn_model_apply(const nn_model *model, const vec *input, vec *output, bool training);

nn_model *nn_model_train(nn_model *model,
                           const data_points *data_x, slice x_sly,
                           const data_points *data_trg, slice trg_sly,
                           const vec *data_weight,
                           slice index_sly,
                           IND_TYP batch_size,
                           int nbr_epochs,
                           bool shuffle,
                           nn_optim *optimizer,
                           const nn_loss loss);

FLT_TYP nn_model_eval(const nn_model *model,
                      const data_points *data_x, slice x_sly,
                      const data_points *data_trg, slice trg_sly,
                      const vec *data_weight,
                      slice index_sly,
                      const nn_loss loss,
                      bool classification);

char *nn_model_to_str(const nn_model *model, char *string);
void nn_model_print(const nn_model *model);

size_t nn_model_nbr_param(const nn_model *model);

size_t nn_model_serial_size(const nn_model *model);
// returns a pointer to the byte after the last written byte
uint8_t *nn_model_serialize(const nn_model *model, uint8_t *byte_arr);
// returns a pointer to the byte after the last read byte
const uint8_t *nn_model_deserialize(nn_model *model, const uint8_t *byte_arr);

void nn_model_save(const nn_model *model, const char *file_path);
nn_model *nn_model_load(nn_model *model, const char *file_path);

// Merges identically structured models
// int nn_model_merge(nn_model *result, const nn_model *model_arr[], FLT_TYP weight_arr[]);
