#include "nn_model_t.h"

#include <string.h>

const nn_model_t nn_model_NULL =
    {.capacity = 0, .nbr_layers = 0, .layer = NULL, .weight = NULL, .bias = NULL};

nn_model_t *nn_model_construct(nn_model_t *model, int capacity)
{
    assert(model);
    assert(capacity > 0);
    if (memcmp(model, &nn_model_NULL, sizeof(nn_model_t)) != 0)
    {
#ifdef DEBUG
        fprintf(stderr, "nn_model_construct: model is not NULL!");
#endif
        return model;
    }
    model->capacity = capacity;
    model->layer = (nn_layer_t *)calloc(capacity, sizeof(nn_layer_t));
    assert(model->layer);
    model->weight = (mat_t *)calloc(capacity, sizeof(mat_t));
    assert(model->weight);
    model->bias = (vec_t *)calloc(capacity, sizeof(vec_t));
    assert(model->bias);
    model->nbr_layers = 0;
    return model;
}

void nn_model_destruct(nn_model_t *model)
{
    assert(model);
    if (memcmp(model, &nn_model_NULL, sizeof(nn_model_t)))
    {
        for (int l = 0; l < model->nbr_layers; l++)
        {
            mat_destruct(model->weight + l);
            vec_destruct(model->bias + l);
        }
        free(model->layer);
        free(model->weight);
        free(model->bias);
    }
    *model = nn_model_NULL;
}

nn_model_t *nn_model_add(nn_model_t *model, const nn_layer_t *layer)
{
    assert(model);
    assert(layer);
    if (memcmp(layer, &nn_layer_NULL, sizeof(nn_layer_t) == 0))
        return model;
    if (model->nbr_layers == model->capacity)
        return model; // or resize

    model->layer[model->nbr_layers] = *layer;
    mat_construct(model->weight + model->nbr_layers, layer->out_sz, layer->inp_sz);
    vec_construct(model->bias + model->nbr_layers, layer->out_sz);
    model->nbr_layers++;
    return model;
}

nn_model_t *nn_model_remove(nn_model_t *model, int layer_index)
{
    assert(model);
    if (layer_index < 0)
        layer_index = model->nbr_layers + layer_index;
    if (layer_index >= model->nbr_layers || layer_index < 0)
        return model;

    model->nbr_layers--;
    mat_destruct(model->weight + layer_index);
    vec_destruct(model->bias + layer_index);
    memmove(model->layer + layer_index, model->layer + layer_index + 1,
            (model->nbr_layers - layer_index) * sizeof(nn_layer_t));
    memmove(model->weight + layer_index, model->weight + layer_index + 1,
            (model->nbr_layers - layer_index) * sizeof(mat_t));
    memmove(model->bias + layer_index, model->bias + layer_index + 1,
            (model->nbr_layers - layer_index) * sizeof(vec_t));
    return model;
}

vec_t *nn_model_eval(nn_model_t *model, const vec_t *input, vec_t *output)
{
    assert(model);
    if (model->nbr_layers == 0)
        return output;
    assert(input);
    assert(output);
    assert(input->d == model->layer->inp_sz);
    assert(output->d == model->layer[model->nbr_layers-1].out_sz);
    mat_dot_vec(output,  ,input);
    vec_addto(output, );
    return layer->act(output, output);
}