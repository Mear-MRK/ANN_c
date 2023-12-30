#include "nn_model_intern.h"

#include <string.h>
#include <assert.h>
#include <stdlib.h>

nn_model_intern *nn_model_intern_construct(nn_model_intern *intern, int layer_capacity, IND_TYP inp_size)
{
    assert(intern);
    assert(layer_capacity > 0);

    intern->d_w = (mat *)calloc(layer_capacity, sizeof(mat));
    assert(intern->d_w);
    intern->d_b = (vec *)calloc(layer_capacity, sizeof(vec));
    assert(intern->d_b);
    intern->a_mask = (vec *)calloc(layer_capacity, sizeof(vec));
    assert(intern->a_mask);
    intern->s = (vec *)calloc(layer_capacity, sizeof(vec));
    assert(intern->s);
    intern->a = (vec *)calloc(layer_capacity, sizeof(vec));
    assert(intern->a);
    intern->nbr_layers = 0;
    intern->a_inp = vec_NULL;
    vec_construct(&intern->a_inp, inp_size);
    return intern;
}

void nn_model_intern_destruct(nn_model_intern *intern)
{
    for (int l = 0; l < intern->nbr_layers; l++)
    {
        mat_destruct(intern->d_w + l);
        vec_destruct(intern->d_b + l);
        vec_destruct(intern->a_mask + l);
        vec_destruct(intern->s + l);
        vec_destruct(intern->a + l);
    }
    free(intern->d_w);
    free(intern->d_b);
    free(intern->a_mask);
    vec_destruct(&intern->a_inp);
    free(intern->s);
    free(intern->a);
    memset(intern, 0, sizeof(nn_model_intern));
}

nn_model_intern *nn_model_intern_add(nn_model_intern *intern, const nn_layer *layer, IND_TYP inp_size)
{
    assert(intern);
    assert(layer);
    mat_construct(intern->d_w + intern->nbr_layers, layer->out_sz, inp_size);
    vec_construct(intern->d_b + intern->nbr_layers, layer->out_sz);
    vec_construct(intern->a_mask + intern->nbr_layers, inp_size);
    vec_construct(intern->s + intern->nbr_layers, layer->out_sz);
    vec_construct(intern->a + intern->nbr_layers, layer->out_sz);
    intern->nbr_layers++;
    return intern;
}

nn_model_intern *nn_model_intern_remove(nn_model_intern *intern, int layer_index)
{
    assert(intern);
    assert(layer_index >= 0 && layer_index < intern->nbr_layers);
    mat_destruct(intern->d_w + layer_index);
    vec_destruct(intern->d_b + layer_index);
    vec_destruct(intern->a_mask + layer_index);
    vec_destruct(intern->s + layer_index);
    vec_destruct(intern->a + layer_index);
    int nsz_mv = intern->nbr_layers - layer_index;
    memmove(intern->d_w + layer_index, intern->d_w + layer_index + 1, nsz_mv * sizeof(mat));
    memmove(intern->d_b + layer_index, intern->d_b + layer_index + 1, nsz_mv * sizeof(vec));
    memmove(intern->a_mask + layer_index, intern->a_mask + layer_index + 1, nsz_mv * sizeof(vec));
    memmove(intern->s + layer_index, intern->s + layer_index + 1, nsz_mv * sizeof(vec));
    memmove(intern->a + layer_index, intern->a + layer_index + 1, nsz_mv * sizeof(vec));
    return intern;
}

static inline IND_TYP *lin_init(IND_TYP *ind, IND_TYP size)
{
    assert(size > 0);
    assert(ind);

    for (IND_TYP i = 0; i < size; i++)
        ind[i] = i;

    return ind;
}

static inline IND_TYP *shuffle_choose_n(IND_TYP *ind, IND_TYP size, IND_TYP n, uint32_t (*rnd_gen)(void))
{
    assert(n <= size && size > 0);
    assert(ind);
    for (IND_TYP i = 0; i < n; i++)
    {
        IND_TYP j = i + rnd_gen() % (size - i);
        if (i != j)
        {
            IND_TYP tmp = ind[i];
            ind[i] = ind[j];
            ind[j] = tmp;
        }
    }
    return ind;
}

void nn_model_reset_gradients(nn_model_intern *intern)
{
    assert(intern);
    for (int l = 0; l < intern->nbr_layers; l++)
    {
        mat_fill_zero(intern->d_w + l);
        vec_fill_zero(intern->d_b + l);
    }
}
