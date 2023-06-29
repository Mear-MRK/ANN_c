#ifndef NN_MODEL_T_H_INCLUDED
#define NN_MODEL_T_H_INCLUDED 1

#include "nn_layer_t.h"

typedef struct nn_model_struct
{
    int capacity;
    int nbr_layers;
    nn_layer_t layer[];
    mat_t weight[];
    vec_t bias[];
} nn_model_t;

extern const nn_model_t nn_model_NULL;

nn_model_t *nn_model_construct(nn_model_t *model, int capacity);
void nn_model_destruct(nn_model_t *model);

nn_model_t *nn_model_init_rnd(nn_madel *model, FLT_TYPE (*rnd_gen)(void));
nn_model_t *nn_model_init_copy(nn_model_t *model, const nn_model_t *src_model);

nn_model_t *nn_model_add(nn_model_t *model, const nn_layer_t *layer);
nn_model_t *nn_model_remove(nn_model_t *model, int layer_index);

vec_t *nn_model_eval(nn_model_t *model, const vec_t *input, vec_t *output);

nn_model_t *nn_model_train(nn_model_t *, const mat_t *data, const vec_t *label);

nn_model_t *nn_model_load(nn_model_t *model, const char *file_path);
void nn_model_save(const nn_model_t *model, const char *file_path);

#endif /* NN_MODEL_T_H_INCLUDED */