#include "nn_model.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#include "rnd.h"
#include "log.h"

const nn_model nn_model_NULL =
    {.input_size = 0,
     .ouput_size = 0,
     .layer_capacity = 0,
     .nbr_layers = 0,
     .max_width = 0,
     .layer = NULL,
     .weight = NULL,
     .bias = NULL,
     .intern = nn_model_intern_NULL};

bool nn_model_is_null(const nn_model *model)
{
    return memcmp(model, &nn_model_NULL, sizeof(nn_model)) == 0;
}

nn_model *nn_model_construct(nn_model *model, int layer_capacity, IND_TYP input_size)
{
    assert(model);
    assert(layer_capacity > 0);
    assert(input_size > 0);

    if (!model)
    {
        log_msg(LOG_ERR, "nn_model_construct: model is pointing to NULL!");
        return NULL;
    }
    if (!nn_model_is_null(model))
    {
        log_msg(LOG_ERR, "nn_model_construct: model is not nn_model_NULL!");
        return model;
    }
    if (layer_capacity <= 0)
    {
        log_msg(LOG_ERR, "nn_model_construct: layer_capacity must be > 0.");
        return NULL;
    }
    if (input_size <= 0)
    {
        log_msg(LOG_ERR, "nn_model_construct: input_size must be > 0.");
        return NULL;
    }
    model->layer_capacity = layer_capacity;
    model->ouput_size = 0;
    model->layer = (nn_layer *)calloc(layer_capacity, sizeof(nn_layer));
    assert(model->layer);
    model->weight = (mat *)calloc(layer_capacity, sizeof(mat));
    assert(model->weight);
    model->bias = (vec *)calloc(layer_capacity, sizeof(vec));
    assert(model->bias);
    model->input_size = input_size;
    model->nbr_layers = 0;
    model->max_width = 0;
    nn_model_intern_construct(&model->intern, layer_capacity, input_size);
    return model;
}

void nn_model_destruct(nn_model *model)
{
    assert(model);
    if (!nn_model_is_null(model))
    {
        for (int l = 0; l < model->nbr_layers; l++)
        {
            mat_destruct(model->weight + l);
            vec_destruct(model->bias + l);
        }
        nn_model_intern_destruct(&model->intern);
        free(model->layer);
        free(model->weight);
        free(model->bias);
    }
    *model = nn_model_NULL;
}

nn_model *nn_model_init_uniform_rnd(nn_model *model, FLT_TYP amp, FLT_TYP mean)
{
    assert(model);

    flt_rnd_param param;
    param.amp = amp;
    param.mean = mean;

    for (int l = 0; l < model->nbr_layers; l++)
    {
        mat_fill_gen(model->weight + l, uniform_flt_rnd, &param);
        vec_fill_gen(model->bias + l, uniform_flt_rnd, &param);
    }
    return model;
}

nn_model *nn_model_append(nn_model *model, const nn_layer *layer)
{
    assert(model);
    assert(layer);
    if (nn_layer_is_null(layer))
    {
        log_msg(LOG_WRN, "nn_model_append: the layer is nn_layer_NULL! nothing appended.");
        return model;
    }
    if (model->nbr_layers == model->layer_capacity)
    {
        log_msg(LOG_WRN, "nn_model_append: layer_capacity of the model has been reached; nothing appended.");
        return model; // or resize
    }
    // makes copy of layer
    model->layer[model->nbr_layers] = *layer;
    int inp_size = (model->nbr_layers == 0) ? model->input_size : model->ouput_size;
    mat_construct(model->weight + model->nbr_layers, layer->out_sz, inp_size);
    vec_construct(model->bias + model->nbr_layers, layer->out_sz);
    nn_model_intern_add(&model->intern, layer, inp_size);
    model->ouput_size = layer->out_sz;
    model->nbr_layers++;
    if (layer->out_sz > model->max_width)
        model->max_width = layer->out_sz;
    return model;
}

nn_model *nn_model_remove(nn_model *model, int layer_index)
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
            (model->nbr_layers - layer_index) * sizeof(nn_layer));
    memmove(model->weight + layer_index, model->weight + layer_index + 1,
            (model->nbr_layers - layer_index) * sizeof(mat));
    memmove(model->bias + layer_index, model->bias + layer_index + 1,
            (model->nbr_layers - layer_index) * sizeof(vec));
    nn_model_intern_remove(&model->intern, layer_index);

    model->max_width = 0;
    for (int l = 0; l < model->nbr_layers; l++)
        if (model->layer[l].out_sz > model->max_width)
            model->max_width = model->layer[l].out_sz;
    if(model->nbr_layers > 0)
        model->ouput_size = model->ouput_size;
    else {
        model->ouput_size = 0;
    }

    return model;
}

static void nn_model_dropping_out(nn_model *model)
{
    nn_layer *layer = model->layer;
    nn_model_intern *intern = &model->intern;

    for (int l = 0; l < model->nbr_layers; l++)
    {
        FLT_TYP drp = layer[l].dropout;
        if (drp != 0)
            for (IND_TYP u = 0; u < intern->a_mask[l].d; u++)
                *vec_at(intern->a_mask + l, u) = (FLT_TYP)(uniform_flt_rnd(NULL) >= drp);
    }
}

vec *nn_model_apply(const nn_model *model, const vec *input, vec *output, bool training)
{
    assert(model);
    if (model->nbr_layers == 0)
        return output;
    assert(vec_is_valid(input));
    assert(vec_is_valid(output));
    assert(input->d == model->input_size);
    assert(output->d == model->ouput_size);

    vec *a_inp = &model->intern.a_inp;
    vec *s = model->intern.s;
    vec *a = model->intern.a;
    vec *a_mask = model->intern.a_mask;

    mat *w = model->weight;
    vec *b = model->bias;

    nn_layer *layer = model->layer;

    if (layer->dropout && training)
        vec_mul(a_inp, a_mask, input);
    else
        vec_assign(a_inp, input);
    mat_dot_vec(s, w, a_inp);
    vec_addto(s, b);
    layer->activ.func(a, s);
    for (int l = 1; l < model->nbr_layers; l++)
    {
        if (layer[l].dropout && training)
            vec_mulby(a + l - 1, a_mask + l);
        mat_dot_vec(s + l, w + l, a + l - 1);
        vec_addto(s + l, b + l);
        layer[l].activ.func(a + l, s + l);
    }
    vec_assign(output, a + model->nbr_layers - 1);
    return output;
}

static inline void nn_model_backprop(nn_model *model, const vec *loss_drv,
                                     vec *buff_1, vec *buff_2)
{
    assert(model);
    assert(vec_is_valid(loss_drv));
    // assert(data_x);
    assert(model->nbr_layers > 0);
    assert(vec_is_valid(buff_1));
    assert(vec_is_valid(buff_2));

    nn_layer *layer = model->layer;
    mat *w = model->weight;
    vec *a = model->intern.a;
    vec *s = model->intern.s;
    vec *a_m = model->intern.a_mask;

    buff_1->d = loss_drv->d;
    vec_assign(buff_1, loss_drv);

    for (int l = model->nbr_layers - 1; l >= 0; l--)
    {
        buff_2->d = a[l].d;
        layer[l].activ.deriv(buff_2, s + l, a + l);
        if (l + 1 != model->nbr_layers && layer[l + 1].dropout)
            vec_mulby(buff_2, a_m + l + 1);
        vec_mulby(buff_2, buff_1);
        const vec *tmp;
        if (l != 0)
        {
            buff_1->d = w[l].d2;
            vec_dot_mat(buff_1, buff_2, w + l);
            tmp = a + l - 1;
        }
        else
        {
            tmp = &model->intern.a_inp; // data_x;
        }
        mat_update_outer(model->intern.d_w + l, 1 / (1 - layer[l].dropout), buff_2, tmp);
        vec_update(model->intern.d_b + l, 1, buff_2);
    }
}

static inline void init_ind(IND_TYP *ind, IND_TYP size)
{
    for (IND_TYP i = 0; i < size; i++)
        ind[i] = i;
}

static inline void shuffle_ind(IND_TYP *ind, IND_TYP size, uint64_t rnd(void))
{
    for (IND_TYP i = 0; i < size - 1; i++)
    {
        IND_TYP j = rnd() % (size - i) + i;
        assert(j >= 0 && j < size);
        if (j != i)
        {
            IND_TYP tmp = ind[i];
            ind[i] = ind[j];
            ind[j] = tmp;
        }
    }
}

nn_model *nn_model_train(nn_model *model,
                           const data_points *data_x, slice x_sly,
                           const data_points *data_trg, slice trg_sly,
                           const vec *data_weight,
                           slice index_sly,
                           IND_TYP batch_size,
                           int nbr_epochs,
                           bool shuffle,
                           nn_optim *optimizer,
                           const nn_loss loss)
{
    assert(model);
    assert(data_points_is_valid(data_x));
    assert(data_points_is_valid(data_trg));
    assert(slice_is_valid(&x_sly));
    assert(slice_is_valid(&trg_sly));
    assert(slice_is_valid(&index_sly));
    assert(batch_size >= 0 && nbr_epochs > 0);

    slice_regulate(&x_sly, data_x->width);
    slice_regulate(&trg_sly, data_trg->width);
    slice_regulate(&index_sly, data_x->nbr_points);

    assert(model->input_size == x_sly.len);
    assert(model->ouput_size == trg_sly.len);
    assert(!data_weight || data_weight->d == index_sly.len);

    if (batch_size == 0)
        batch_size = index_sly.len;

    if (model->nbr_layers == 0 || index_sly.len == 0 || nbr_epochs <= 0 || batch_size <= 0)
    {
        log_msg(LOG_WRN, "nn_model_train: the model can't be trained with the given params!");
        return model;
    }
    if (model->input_size != x_sly.len || model->ouput_size != trg_sly.len || (data_weight && data_weight->d != index_sly.len))
    {
        log_msg(LOG_WRN, "nn_model_train: mismatch sizes! nothing trained.");
        return model;
    }

    vec *buff_1 = vec_new(model->max_width);
    vec *buff_2 = vec_new(model->max_width);
    vec *output = vec_new(model->ouput_size);
    vec *loss_drv = vec_new(model->ouput_size);
    vec feat = vec_NULL, lbl = vec_NULL;
    IND_TYP nbr_data = index_sly.len;
    IND_TYP nbr_batch = nbr_data / batch_size;
    IND_TYP nbr_rem_data = nbr_data - nbr_batch * batch_size;

    IND_TYP *ind = (IND_TYP *)calloc(nbr_data, sizeof(IND_TYP));
    assert(ind);
    init_ind(ind, nbr_data);

    log_msg(LOG_INF, "nn_model_train: training began.");
    for (IND_TYP epoch = 0; epoch < nbr_epochs; epoch++)
    {
        if (shuffle)
            shuffle_ind(ind, nbr_data, UINT_RND_GEN);
        IND_TYP i = 0;
        for (IND_TYP batch = 0; batch < nbr_batch; batch++)
        {

            nn_model_dropping_out(model);
            nn_model_reset_gradients(&model->intern);
            for (IND_TYP b_i = 0; b_i < batch_size; b_i++, i++)
            {
                IND_TYP k = slice_index(&index_sly, ind[i]);
                data_points_at(data_x, &feat, k, &x_sly);
                nn_model_apply(model, &feat, output, true);
                data_points_at(data_trg, &lbl, k, &trg_sly);
                loss.deriv(loss_drv, &lbl, output);
                if (data_weight)
                    vec_scale(loss_drv, *vec_at(data_weight, k));
                nn_model_backprop(model, loss_drv, buff_1, buff_2);
            }
            nn_optim_update_model(optimizer, model);
        }
        if (nbr_rem_data > 0)
        {
            nn_model_dropping_out(model);
            nn_model_reset_gradients(&model->intern);
            for (; i < nbr_data; i++)
            {
                IND_TYP k = slice_index(&index_sly, ind[i]);
                data_points_at(data_x, &feat, k, &x_sly);
                nn_model_apply(model, &feat, output, true);
                data_points_at(data_trg, &lbl, k, &trg_sly);
                loss.deriv(loss_drv, &lbl, output);
                if (data_weight)
                    vec_scale(loss_drv, *vec_at(data_weight, k));
                nn_model_backprop(model, loss_drv, buff_1, buff_2);
            }
            nn_optim_update_model(optimizer, model);
        }
        log_msg(LOG_DBG, "nn_model_train: epoch  %d/%d finished.", epoch + 1, nbr_epochs);
    }
    log_msg(LOG_INF, "nn_model_train: training ended.");

    free(ind);
    vec_del(loss_drv);
    vec_del(output);
    vec_del(buff_2);
    vec_del(buff_1);
    vec_destruct(&feat);
    vec_destruct(&lbl);
    return model;
}

FLT_TYP nn_model_eval(const nn_model *model, const data_points *data_x, slice x_sly,
                      const data_points *data_trg, slice trg_sly,
                      const vec *data_weight, slice index_sly,
                      const nn_loss loss, bool classification)
{
    assert(model);
    assert(data_points_is_valid(data_x));
    assert(data_points_is_valid(data_trg));
    assert(slice_is_valid(&x_sly));
    assert(slice_is_valid(&trg_sly));
    assert(slice_is_valid(&index_sly));

    slice_regulate(&x_sly, data_x->width);
    slice_regulate(&trg_sly, data_trg->width);
    slice_regulate(&index_sly, data_x->nbr_points);

    assert(model->input_size == x_sly.len);
    assert(model->ouput_size == trg_sly.len);
    assert(!data_weight || data_weight->d == index_sly.len);

    FLT_TYP loss_value = 0;
    FLT_TYP trg_nrm = 0;
    vec inp = vec_NULL, trg = vec_NULL;
    vec *out = vec_new(trg_sly.len);
    vec *buf = vec_new(trg_sly.len);
    for (int i = 0; i < index_sly.len; i++)
    {
        IND_TYP k = slice_index(&index_sly, i);
        data_points_at(data_x, &inp, k, &x_sly);
        nn_model_apply(model, &inp, out, false);
        data_points_at(data_trg, &trg, k, &trg_sly);
        FLT_TYP w = 1;
        if (data_weight)
            w = *vec_at(data_weight, k);
        if (!classification)
        {
            trg_nrm += w * vec_norm_2(&trg);
            loss_value += w * loss.func(&trg, out, buf);
        }
        else
        {
            trg_nrm += w;
            IND_TYP im = vec_argmax(out);
            loss_value += w * (1 - *vec_at(&trg, im));
        }
    }
    vec_del(buf);
    vec_del(out);
    vec_destruct(&inp);
    vec_destruct(&trg);
    return loss_value / trg_nrm;
}

char *nn_model_to_str(const nn_model *model, char *string)
{
    assert(model);
    assert(string);
    char buff[128];
    buff[0] = 0;
    sprintf(string, "nn_model: input size %ld, nbr of layers %d, max width %ld, layer_capacity %d; layers:\n",
            model->input_size, model->nbr_layers, model->max_width, model->layer_capacity);
    for (int l = 0; l < model->nbr_layers; l++)
    {
        sprintf(buff, "%3d ", l);
        strcat(string, buff);
        strcat(string, nn_layer_to_str(model->layer + l, buff));
        strcat(string, "\n");
    }
    return string;
}

void nn_model_print(const nn_model *model)
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

size_t nn_model_nbr_param(const nn_model *model)
{
    assert(model);
    if (model->nbr_layers <= 0)
        return 0;
    size_t nbr_param = (model->input_size + 1) * model->layer[0].out_sz;
    for (int l = 1; l < model->nbr_layers; l++)
    {
        nbr_param += model->layer[l].out_sz * (1 + model->layer[l - 1].out_sz);
    }
    return nbr_param;
}

static inline uint8_t *wrt2byt(const void *obj, size_t sz, uint8_t *bytes)
{
    assert(obj);
    assert(bytes);
    assert(sz);
    memcpy(bytes, obj, sz);
    return bytes + sz;
}

size_t nn_model_serial_size(const nn_model *model)
{
    assert(model);
    size_t size = 0;
    size += sizeof(size) +
            sizeof(model->layer_capacity) + sizeof(model->input_size) +
            sizeof(model->nbr_layers) + sizeof(model->max_width);
    for (IND_TYP l = 0; l < model->nbr_layers; l++)
    {
        size += nn_layer_serial_size(model->layer + l);
        size += mat_serial_size(model->weight + l);
        size += vec_serial_size(model->bias + l);
    }
    return size;
}

uint8_t *nn_model_serialize(const nn_model *model, uint8_t *byte_arr)
{
    assert(model);
    assert(byte_arr);
    size_t sz = nn_model_serial_size(model);
    byte_arr = wrt2byt(&sz, sizeof(size_t), byte_arr);
    byte_arr = wrt2byt(&model->layer_capacity, sizeof(model->layer_capacity), byte_arr);
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
    assert(bytes);
    assert(sz);
    if (!obj)
        return bytes + sz;
    memcpy(obj, bytes, sz);
    return bytes + sz;
}

const uint8_t *nn_model_deserialize(nn_model *model, const uint8_t *byte_arr)
{
    assert(model);
    assert(byte_arr);
    assert(nn_model_is_null(model));
    int cap = 0;
    IND_TYP inp_sz = 0;
    byte_arr = rd_byt(NULL, sizeof(size_t), byte_arr);
    byte_arr = rd_byt(&cap, sizeof(model->layer_capacity), byte_arr);
    byte_arr = rd_byt(&inp_sz, sizeof(model->input_size), byte_arr);
    nn_model_construct(model, cap, inp_sz);
    byte_arr = rd_byt(&model->nbr_layers, sizeof(model->nbr_layers), byte_arr);
    byte_arr = rd_byt(&model->max_width, sizeof(model->max_width), byte_arr);
    nn_model_intern_construct(&model->intern, cap, inp_sz);
    for (IND_TYP l = 0; l < model->nbr_layers; l++)
    {
        byte_arr = nn_layer_deserialize(model->layer + l, byte_arr);
        byte_arr = mat_deserialize(model->weight + l, byte_arr);
        byte_arr = vec_deserialize(model->bias + l, byte_arr);
        IND_TYP ly_inp_sz = (l != 0) ? model->layer[l - 1].out_sz : inp_sz;
        nn_model_intern_add(&model->intern, model->layer + l, ly_inp_sz);
    }
    return byte_arr;
}

void nn_model_save(const nn_model *model, const char *file_path)
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
    size_t wr_sz = fwrite(byte_arr, 1, size, file);
    if (wr_sz != size)
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

nn_model *nn_model_load(nn_model *model, const char *file_path)
{
    FILE *file = fopen(file_path, "rb");
    if (!file)
    {
        perror("nn_model_load: can't open the file!");
        exit(-2);
    }
    size_t size = 0;
    fread(&size, sizeof(size), 1, file);
    uint8_t *byte_arr = malloc(size);
    assert(byte_arr);
    size_t sz = fread(byte_arr, 1, size, file);
    if (sz != size)
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
// size chunk_size = 4096;
// size total_size = 0;
// size bytes_read = 0;
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