#include "nn_model.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

const nn_model_t nn_model_NULL =
    {.capacity = 0, .nbr_layers = 0, .max_width = 0, .layer = NULL, .weight = NULL, .bias = NULL, .d_w = NULL, .d_b = NULL, .s = NULL, .a = NULL};

bool nn_model_is_null(const nn_model_t *model)
{
    return memcmp(model, &nn_model_NULL, sizeof(nn_model_t)) == 0;
}

nn_model_t *nn_model_construct(nn_model_t *model, int capacity, IND_TYP input_size)
{
    assert(model);
    if (!nn_model_is_null(model))
    {
        fprintf(stderr, "nn_model_construct: model is not NULL!");
        return model;
    }
    if (capacity <= 0)
    {
        fprintf(stderr, "nn_model_construct: capacity must be > 0.");
        return NULL;
    }
    if (input_size <= 0)
    {
        fprintf(stderr, "nn_model_construct: input_size must be > 0.");
        return model;
    }
    model->capacity = capacity;
    model->layer = (nn_layer_t *)calloc(capacity, sizeof(nn_layer_t));
    assert(model->layer);
    model->weight = (mat_t *)calloc(capacity, sizeof(mat_t));
    assert(model->weight);
    model->bias = (vec_t *)calloc(capacity, sizeof(vec_t));
    assert(model->bias);
    model->d_w = (mat_t *)calloc(capacity, sizeof(mat_t));
    assert(model->d_w);
    model->d_b = (vec_t *)calloc(capacity, sizeof(vec_t));
    assert(model->d_b);
    model->s = (vec_t *)calloc(capacity, sizeof(vec_t));
    assert(model->s);
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
            mat_destruct(model->d_w + l);
            vec_destruct(model->d_b + l);
            vec_destruct(model->s + l);
            vec_destruct(model->a + l);
        }
        free(model->layer);
        free(model->weight);
        free(model->bias);
        free(model->d_w);
        free(model->d_b);
        free(model->s);
        free(model->a);
    }
    *model = nn_model_NULL;
}

nn_model_t *nn_model_init_rnd(nn_model_t *model, FLT_TYP (*rnd_gen)(void))
{
    assert(model);
    assert(rnd_gen);
    for (int l = 0; l < model->nbr_layers; l++)
    {
        mat_fill_rnd(model->weight + l, rnd_gen);
        vec_fill_rnd(model->bias + l, rnd_gen);
    }
    return model;
}

nn_model_t *nn_model_add(nn_model_t *model, const nn_layer_t *layer)
{
    assert(model);
    assert(layer);
    if (memcmp(layer, &nn_layer_NULL, sizeof(nn_layer_t) == 0))
        return model;
    if (model->nbr_layers == model->capacity)
        return model; // or resize
    // makes copy of layer
    model->layer[model->nbr_layers] = *layer;
    int inp_size = (model->nbr_layers == 0) ? model->input_size : model->layer[model->nbr_layers - 1].out_sz;
    mat_construct(model->weight + model->nbr_layers, layer->out_sz, inp_size);
    vec_construct(model->bias + model->nbr_layers, layer->out_sz);
    mat_construct(model->d_w + model->nbr_layers, layer->out_sz, inp_size);
    vec_construct(model->d_b + model->nbr_layers, layer->out_sz);
    vec_construct(model->s + model->nbr_layers, layer->out_sz);
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
    mat_destruct(model->d_w + layer_index);
    vec_destruct(model->d_b + layer_index);
    vec_destruct(model->s + layer_index);
    vec_destruct(model->a + layer_index);
    memmove(model->layer + layer_index, model->layer + layer_index + 1,
            (model->nbr_layers - layer_index) * sizeof(nn_layer_t));
    memmove(model->weight + layer_index, model->weight + layer_index + 1,
            (model->nbr_layers - layer_index) * sizeof(mat_t));
    memmove(model->bias + layer_index, model->bias + layer_index + 1,
            (model->nbr_layers - layer_index) * sizeof(vec_t));
    memmove(model->d_w + layer_index, model->d_w + layer_index + 1,
            (model->nbr_layers - layer_index) * sizeof(mat_t));
    memmove(model->d_b + layer_index, model->d_b + layer_index + 1,
            (model->nbr_layers - layer_index) * sizeof(vec_t));
    memmove(model->s + layer_index, model->s + layer_index + 1,
            (model->nbr_layers - layer_index) * sizeof(vec_t));
    memmove(model->a + layer_index, model->a + layer_index + 1,
            (model->nbr_layers - layer_index) * sizeof(vec_t));

    model->max_width = 0;
    for (int l = 0; l < model->nbr_layers; l++)
        if (model->layer[l].out_sz > model->max_width)
            model->max_width = model->layer[l].out_sz;

    return model;
}

vec_t *nn_model_apply(const nn_model_t *model, const vec_t *input, vec_t *output)
{
    assert(model);
    if (model->nbr_layers == 0)
        return output;
    assert(input);
    assert(input->arr);
    assert(output);
    assert(output->arr);
    assert(input->size == model->input_size);
    assert(output->size == model->layer[model->nbr_layers - 1].out_sz);

    mat_dot_vec(model->s, model->weight, input);
    vec_addto(model->s, model->bias);
    model->layer->activ.func(model->a, model->s);
    for (int l = 1; l < model->nbr_layers; l++)
    {
        mat_dot_vec(model->s + l, model->weight + l, model->a + l - 1);
        vec_addto(model->s + l, model->bias + l);
        model->layer[l].activ.func(model->a + l, model->s + l);
    }
    vec_assign(output, model->a + model->nbr_layers - 1);
    return output;
}

void nn_model_backprop(nn_model_t *model, const vec_t *err_drv,
                       const vec_t *data_x,
                       vec_t *buff_1, vec_t *buff_2)
{
    assert(model);
    assert(err_drv);
    assert(data_x);
    assert(model->nbr_layers > 0);
    assert(buff_1);
    assert(buff_2);

    buff_1->size = err_drv->size;
    vec_assign(buff_1, err_drv);

    for (int l = model->nbr_layers - 1; l >= 0; l--)
    {
        buff_2->size = model->a[l].size;
        model->layer[l].activ.deriv(buff_2, model->s + l, model->a + l);
        vec_mulby(buff_2, buff_1);
        const vec_t *tmp;
        if (l != 0)
        {
            buff_1->size = model->weight[l].d2;
            vec_dot_mat(buff_1, buff_2, model->weight + l);
            tmp = model->a + l - 1;
        }
        else
        {
            tmp = data_x;
        }
        mat_update_outer(model->d_w + l, 1, buff_2, tmp);
        vec_update(model->d_b + l, 1, buff_2);
    }
}

static inline void nn_model_reset_d_w_d_b(nn_model_t *model)
{
    assert(model);
    for (int l = 0; l < model->nbr_layers; l++)
    {
        mat_fill_zero(model->d_w + l);
        vec_fill_zero(model->d_b + l);
    }
}

static inline void nn_model_update_w_b(nn_model_t *model, FLT_TYP learning_rate)
{
    assert(learning_rate > 0);
    FLT_TYP alpha = -learning_rate;
    for (int l = 0; l < model->nbr_layers; l++)
    {
        mat_update(model->weight + l, alpha, model->d_w + l);
        vec_update(model->bias + l, alpha, model->d_b + l);
    }
}

static inline void init_ind(IND_TYP *ind, IND_TYP size)
{
    for (IND_TYP i = 0; i < size; i++)
        ind[i] = i;
}

static inline void shuffle_ind(IND_TYP *ind, IND_TYP size, IND_TYP rnd(void))
{
    IND_TYP tmp, j;
    for (IND_TYP i = 0; i < size - 1; i++)
    {
        j = rnd() % (size - i) + i;
        assert(j >= 0 && j < size);
        if (j != i)
        {
            tmp = ind[i];
            ind[i] = ind[j];
            ind[j] = tmp;
        }
    }
}

// 30-bit random
static inline IND_TYP rnd(void)
{
    IND_TYP r = rand() * RAND_MAX + rand();
    return (r >= 0) ? r : -r;
}

nn_model_t *nn_model_train(nn_model_t *model,
                           const mat_t *data_x,
                           const mat_t *data_trg,
                           int batch_size,
                           int nbr_epochs,
                           bool shuffle,
                           FLT_TYP learning_rate,
                           const nn_err_t err,
                           nn_model_type type)
{
    assert(model);
    assert(data_x);
    assert(data_trg);
    assert(data_x->d1 == data_trg->d1);
    assert(batch_size > 0 && nbr_epochs > 0);
    if (model->nbr_layers == 0 || data_x->d1 == 0 || data_trg->d1 == 0)
    {
        return model;
    }
    if (nbr_epochs <= 0 || batch_size <= 0)
    {
        return model;
    }
    vec_t *buff_1 = vec_new(model->max_width);
    vec_t *buff_2 = vec_new(model->max_width);
    vec_t *output = vec_new(model->layer[model->nbr_layers - 1].out_sz);
    vec_t *err_drv = vec_new(model->layer[model->nbr_layers - 1].out_sz);
    vec_t feat, lbl;
    IND_TYP nbr_data = data_x->d1;
    IND_TYP nbr_batch = nbr_data / batch_size;
    IND_TYP nbr_rem_data = nbr_data - nbr_batch * batch_size;

    IND_TYP *ind = (IND_TYP *)calloc(nbr_data, sizeof(IND_TYP));
    assert(ind);
    init_ind(ind, nbr_data);
    srand(time(NULL));

    // puts("Training...");
    for (IND_TYP epoch = 0; epoch < nbr_epochs; epoch++)
    {
        if (shuffle)
            shuffle_ind(ind, nbr_data, rnd);
        IND_TYP i = 0;
        for (IND_TYP batch = 0; batch < nbr_batch; batch++)
        {
            nn_model_reset_d_w_d_b(model);
            for (IND_TYP b_i = 0; b_i < batch_size; b_i++, i++)
            {
                vec_init_prealloc(&feat, data_x->arr + ind[i] * data_x->d2, data_x->d2);
                nn_model_apply(model, &feat, output);
                vec_init_prealloc(&lbl, data_trg->arr + ind[i] * data_trg->d2, data_trg->d2);
                err.deriv(err_drv, &lbl, output);
                nn_model_backprop(model, err_drv, &feat, buff_1, buff_2);
            }
            nn_model_update_w_b(model, learning_rate);
        }
        if (nbr_rem_data > 0)
        {
            nn_model_reset_d_w_d_b(model);
            for (; i < nbr_data; i++)
            {
                vec_init_prealloc(&feat, data_x->arr + ind[i] * data_x->d2, data_x->d2);
                nn_model_apply(model, &feat, output);
                vec_init_prealloc(&lbl, data_trg->arr + ind[i] * data_trg->d2, data_trg->d2);
                err.deriv(err_drv, &lbl, output);
                nn_model_backprop(model, err_drv, &feat, buff_1, buff_2);
            }
            nn_model_update_w_b(model, learning_rate);
        }
        // printf("\repoch  %d/%d finished.", epoch + 1, nbr_epochs);
    }
    // puts("\n...done.");

    free(ind);
    vec_del(err_drv);
    vec_del(output);
    vec_del(buff_2);
    vec_del(buff_1);
    return model;
}

FLT_TYP nn_model_eval(const nn_model_t *model, const mat_t *data_x, const mat_t *data_trg,
                      const nn_err_t err)
{
    assert(model);
    assert(data_x);
    assert(data_trg);
    assert(data_x->d1 == data_trg->d1);
    if (model->nbr_layers == 0 || data_x->d1 == 0 || data_trg->d1 == 0)
        return -1;
    FLT_TYP err_value = 0;
    FLT_TYP trg_nrm = 0;
    vec_t inp, trg;
    vec_t *out = vec_new(data_trg->d2);
    vec_t *buf = vec_new(data_trg->d2);
    for (int i = 0; i < data_x->d1; i++)
    {
        vec_init_prealloc(&inp, data_x->arr + i * data_x->d2, data_x->d2);
        nn_model_apply(model, &inp, out);
        vec_init_prealloc(&trg, data_trg->arr + i * data_trg->d2, data_trg->d2);
        trg_nrm += vec_norm_2(&trg);
        err_value += err.func(&trg, out, buf);
    }
    vec_del(buf);
    vec_del(out);
    return err_value / trg_nrm;
}

char *nn_model_to_str(const nn_model_t *model, char *string)
{
    assert(model);
    assert(string);
    char buff[128];
    buff[0] = 0;
    sprintf(string, "nn_model: input size %d, nbr of layers %d, max width %d, capacity %d; layers:\n",
            model->input_size, model->nbr_layers, model->max_width, model->capacity);
    for (int l = 0; l < model->nbr_layers; l++)
    {
        sprintf(buff, "%3d ", l);
        strcat(string, buff);
        strcat(string, nn_layer_to_str(model->layer + l, buff));
        strcat(string, "\n");
    }
    return string;
}

void nn_model_print(const nn_model_t *model)
{
    char str_buff[4096];
    printf(nn_model_to_str(model, str_buff));
    for (int l = 0; l < model->nbr_layers; l++)
    {
        printf("layer %d:\n", l);
        mat_to_str(model->weight + l, str_buff);
        printf("w[%d]:\n%s\n", l, str_buff);
        vec_to_str(model->bias + l, str_buff);
        printf("b[%d]:\n%s\n", l, str_buff);
        // puts("");
    }
}

static inline uint8_t *wrt2byt(const void *obj, size_t sz, uint8_t *bytes)
{
    assert(obj);
    assert(bytes);
    assert(sz);
    memcpy(bytes, obj, sz);
    return bytes + sz;
}

size_t nn_model_serial_size(const nn_model_t *model)
{
    assert(model);
    size_t size = 0;
    size += sizeof(model->capacity) + sizeof(model->input_size) +
            sizeof(model->nbr_layers) + sizeof(model->max_width);
    for (IND_TYP l = 0; l < model->nbr_layers; l++)
    {
        size += nn_layer_serial_size(model->layer + l);
        size += mat_serial_size(model->weight + l);
        size += vec_serial_size(model->bias + l);
    }
    return size;
}

uint8_t *nn_model_serialize(const nn_model_t *model, uint8_t *byte_arr)
{
    assert(model);
    assert(byte_arr);
    byte_arr = wrt2byt(&model->capacity, sizeof(model->capacity), byte_arr);
    byte_arr = wrt2byt(&model->input_size, sizeof(model->input_size), byte_arr);
    byte_arr = wrt2byt(&model->nbr_layers, sizeof(model->nbr_layers), byte_arr);
    byte_arr = wrt2byt(&model->max_width, sizeof(model->max_width), byte_arr);
    for (IND_TYP l = 0; l < model->nbr_layers; l++)
    {
        byte_arr = nn_layer_serialize(model->layer + l, byte_arr);
        byte_arr = mat_serialize(model->weight + l, byte_arr);
        byte_arr = vec_serialize(model->bias + l, byte_arr);
    }
    return byte_arr;
}

static inline const uint8_t *rd_byt(void *obj, size_t sz, const uint8_t *bytes)
{
    assert(obj);
    assert(bytes);
    assert(sz);
    memcpy(obj, bytes, sz);
    return bytes + sz;
}

const uint8_t *nn_model_deserialize(nn_model_t *model, const uint8_t *byte_arr)
{
    assert(model);
    assert(byte_arr);
    assert(nn_model_is_null(model));
    int cap = 0;
    IND_TYP inp_sz = 0;
    byte_arr = rd_byt(&cap, sizeof(model->capacity), byte_arr);
    byte_arr = rd_byt(&inp_sz, sizeof(model->input_size), byte_arr);
    nn_model_construct(model, cap, inp_sz);
    byte_arr = rd_byt(&model->nbr_layers, sizeof(model->nbr_layers), byte_arr);
    byte_arr = rd_byt(&model->max_width, sizeof(model->max_width), byte_arr);
    for (IND_TYP l = 0; l < model->nbr_layers; l++)
    {
        byte_arr = nn_layer_deserialize(model->layer + l, byte_arr);
        byte_arr = mat_deserialize(model->weight + l, byte_arr);
        byte_arr = vec_deserialize(model->bias + l, byte_arr);
        mat_construct(model->d_w + l, model->weight[l].d1, model->weight[l].d2);
        vec_construct(model->d_b + l, model->bias[l].size);
        vec_construct(model->s + l, model->bias[l].size);
        vec_construct(model->a + l, model->bias[l].size);
    }
    return byte_arr;
}

void nn_model_save(const nn_model_t *model, const char *file_path)
{
    FILE *file = fopen(file_path, "wb");
    if (!file)
    {
        perror("nn_model_save: can't open the file!");
        exit(-2);
    }
    size_t size = nn_model_serial_size(model);
    uint8_t *byte_arr = malloc(size);
    assert(byte_arr);
    const uint8_t *ptr = nn_model_serialize(model, byte_arr);
    size_t wr_sz = fwrite(&size, sizeof(size_t), 1, file);
    wr_sz += fwrite(byte_arr, 1, size, file);
    if (wr_sz != size + 1)
    {
        perror("nn_model_save: can't write (completely) to the file!");
        fclose(file);
        free(byte_arr);
        exit(-3);
    }
    fclose(file);
    free(byte_arr);
    assert(size == ((size_t)(ptr - byte_arr)));
}

nn_model_t *nn_model_load(nn_model_t *model, const char *file_path)
{
    FILE *file = fopen(file_path, "rb");
    if (!file)
    {
        perror("nn_model_load: can't open the file!");
        exit(-2);
    }
    size_t size = 0;
    size_t sz = fread(&size, sizeof(size_t), 1, file);
    uint8_t *byte_arr = malloc(size);
    assert(byte_arr);
    sz += fread(byte_arr, 1, size, file);
    if (sz != size + 1)
    {
        perror("nn_model_load: can't read (completely) from the file!");
        fclose(file);
        free(byte_arr);
        exit(-3);
    }
    fclose(file);
    const uint8_t *ptr = nn_model_deserialize(model, byte_arr);
    assert(size == ((size_t)(ptr - byte_arr)));
    free(byte_arr);
    return model;
}

// uint8_t *byte_arr = NULL;
// size_t chunk_size = 4096;
// size_t total_size = 0;
// size_t bytes_read = 0;
// do
// {
//     byte_arr = realloc(byte_arr, total_size + chunk_size);
//     if (byte_arr == NULL)
//     {
//         perror("nn_model_load: Memory allocation failed");
//         fclose(file);
//         exit(EXIT_FAILURE);
//     }
//     bytes_read = fread(byte_arr + total_size, 1, chunk_size, file);
//     total_size += bytes_read;
// } while (bytes_read == chunk_size);