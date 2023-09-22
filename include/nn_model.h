#ifndef NN_MODEL_T_H_INCLUDED
#define NN_MODEL_T_H_INCLUDED 1

#include <stdbool.h>

#include "nn_layer.h"
#include "nn_err.h"

typedef struct nn_model_struct
{
    int capacity;
    IND_TYP input_size;
    int nbr_layers;
    int max_width;
    nn_layer_t *layer;
    mat_t *weight;
    vec_t *bias;
    mat_t *d_w;
    vec_t *d_b;
    vec_t *s;
    vec_t *a;
} nn_model_t;

typedef enum nn_model_type_enum
{
    Regression,
    Classification
} nn_model_type;

extern const nn_model_t nn_model_NULL;

bool nn_model_is_null(const nn_model_t *model);

nn_model_t *nn_model_construct(nn_model_t *model, int capacity, IND_TYP input_size);
void nn_model_destruct(nn_model_t *model);

nn_model_t *nn_model_init_rnd(nn_model_t *model, FLT_TYP (*rnd_gen)(void));
// nn_model_t *nn_model_init_copy(nn_model_t *model, const nn_model_t *src_model);

nn_model_t *nn_model_add(nn_model_t *model, const nn_layer_t *layer);
nn_model_t *nn_model_remove(nn_model_t *model, int layer_index);

vec_t *nn_model_apply(const nn_model_t *model, const vec_t *input, vec_t *output);

nn_model_t *nn_model_train(nn_model_t *model,
                           const mat_t *data_x,
                           const mat_t *data_trg,
                           int batch_size,
                           int nbr_epochs,
                           bool shuffle,
                           FLT_TYP learning_rate,
                           const nn_err_t err,
                           nn_model_type type);

FLT_TYP nn_model_eval(const nn_model_t *model, const mat_t *data_x, const mat_t *data_trg,
 const nn_err_t err);

void nn_model_print(nn_model_t *model);


size_t nn_model_serial_size(const nn_model_t* model);
// returns a pointer to the byte after the last written byte
uint8_t *nn_model_serialize(const nn_model_t *model, uint8_t *byte_arr);
// returns a pointer to the byte after the last read byte
const uint8_t *nn_model_deserialize(nn_model_t *model, const uint8_t *byte_arr);

void nn_model_save(const nn_model_t *model, const char *file_path);
nn_model_t *nn_model_load(nn_model_t *model, const char *file_path);

#endif /* NN_MODEL_T_H_INCLUDED */