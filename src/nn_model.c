#include "nn_model.h"

#include <string.h>

const nn_model_t nn_model_NULL =
    {.capacity = 0, .nbr_layers = 0, .max_width = 0, 
    .layer = NULL, .weight = NULL, .bias = NULL, .a = NULL};

bool nn_model_is_null(const nn_model_t *model)
{
    return memcmp(model, &nn_model_NULL, sizeof(nn_model_t)) == 0;
}

nn_model_t *nn_model_construct(nn_model_t *model, int capacity, int input_size)
{
    assert(model);
    assert(capacity > 0);
    assert(input_size > 0);
    if (!nn_model_is_null(model))
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
    model->a = (vec_t *)calloc(capacity, sizeof(vec_t));
    assert(model->a);
    model->input_size = input_size;
    model->nbr_layers = 0;
    model->max_width = 0;
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
    int inp_size = (model->nbr_layers == 0) ? model->input_size : 
                    model->layer[model->nbr_layers - 1].out_sz;
    mat_construct(model->weight + model->nbr_layers, layer->out_sz, inp_size);
    vec_construct(model->bias + model->nbr_layers, layer->out_sz);
    vec_construct(model->a + model->nbr_layers, layer->out_sz);
    model->nbr_layers++;
    if (layer->out_sz > model->max_width)
        model->max_width = layer->out_sz;
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
    vec_destruct(model->a + layer_index);
    memmove(model->layer + layer_index, model->layer + layer_index + 1,
            (model->nbr_layers - layer_index) * sizeof(nn_layer_t));
    memmove(model->weight + layer_index, model->weight + layer_index + 1,
            (model->nbr_layers - layer_index) * sizeof(mat_t));
    memmove(model->bias + layer_index, model->bias + layer_index + 1,
            (model->nbr_layers - layer_index) * sizeof(vec_t));
    memmove(model->a + layer_index, model->a + layer_index + 1,
            (model->nbr_layers - layer_index) * sizeof(vec_t));

    model->max_width = 0;
    for(int l = 0; l < model->nbr_layers; l++)
        if (model->layer[l].out_sz > model->max_width)
            model->max_width = model->layer[l].out_sz;
    
    return model;
}

vec_t *nn_model_eval(const nn_model_t *model, const vec_t *input, vec_t *output)
{
    assert(model);
    if (model->nbr_layers == 0)
        return output;
    assert(input);
    assert(output);
    assert(input->size == model->input_size);
    assert(output->size == model->layer[model->nbr_layers - 1].out_sz);

    mat_dot_vec(model->a, model->weight, input);
    vec_addto(model->a, model->bias);
    model->layer->activ.act(model->a, model->a);
    for (int l = 1; l < model->nbr_layers; l++)
    {
        mat_dot_vec(model->a + l, model->weight + l, model->a + l - 1);
        vec_addto(model->a + l, model->bias + l);
        model->layer[l].activ.act(model->a + l, model->a + l);
    }
    vec_assign(output, model->a + model->nbr_layers - 1);
    return output;
}

void nn_model_update_backprop(nn_model_t *model, const vec_t *diff, 
                const vec_t *inp_data, FLT_TYP learning_rate, vec_t* buff)
{
    assert(model);
    assert(diff);
    assert(inp_data);
    assert(model->nbr_layers > 0);
    assert(learning_rate > 0);

    for(int l = model->nbr_layers - 1; l > 0; l++)
    {
        
    }
}