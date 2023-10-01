#ifndef NN_MODEL_T_H_INCLUDED
#define NN_MODEL_T_H_INCLUDED 1

#include <stdbool.h>

#include "nn_layer.h"
#include "nn_model_intern.h"

typedef struct nn_model_struct
{
    int capacity;
    IND_TYP input_size;
    int nbr_layers;
    int max_width;
    nn_layer_t *layer;
    mat_t *weight;
    vec_t *bias;
    nn_model_intern_t intern;
} nn_model_t;

extern const nn_model_t nn_model_NULL;

#include "nn_optim.h"
#include "nn_err.h"

bool nn_model_is_null(const nn_model_t *model);

nn_model_t *nn_model_construct(nn_model_t *model, int capacity, IND_TYP input_size);

void nn_model_destruct(nn_model_t *model);

void nn_model_set_rnd_gens(nn_model_t *model, uint32_t (*ui32_rnd)(void), float (*flt_rnd)(void));

nn_model_t *nn_model_init_rnd(nn_model_t *model, FLT_TYP amp, FLT_TYP mean);
// nn_model_t *nn_model_init_copy(nn_model_t *model, const nn_model_t *src_model);

nn_model_t *nn_model_add(nn_model_t *model, const nn_layer_t *layer);
nn_model_t *nn_model_remove(nn_model_t *model, int layer_index);

vec_t *nn_model_apply(const nn_model_t *model, const vec_t *input, vec_t *output, bool training);

nn_model_t *nn_model_train(nn_model_t *model,
                           const mat_t *data_x,
                           const mat_t *data_trg,
                           int batch_size,
                           int nbr_epochs,
                           bool shuffle,
                           nn_optim_t *optimizer,
                           const nn_err_t err);

FLT_TYP nn_model_eval(const nn_model_t *model, const mat_t *data_x, const mat_t *data_trg,
                      const nn_err_t err);

char *nn_model_to_str(const nn_model_t *model, char *string);
void nn_model_print(const nn_model_t *model);

size_t nn_model_serial_size(const nn_model_t *model);
// returns a pointer to the byte after the last written byte
uint8_t *nn_model_serialize(const nn_model_t *model, uint8_t *byte_arr);
// returns a pointer to the byte after the last read byte
const uint8_t *nn_model_deserialize(nn_model_t *model, const uint8_t *byte_arr);

void nn_model_save(const nn_model_t *model, const char *file_path);
nn_model_t *nn_model_load(nn_model_t *model, const char *file_path);

// Merges identically structured models
// int nn_model_merge(nn_model_t *result, const nn_model_t *model_arr[], FLT_TYP weight_arr[]);

#endif /* NN_MODEL_T_H_INCLUDED */