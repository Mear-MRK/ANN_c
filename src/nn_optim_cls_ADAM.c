#include "nn_optim_cls_ADAM.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "nn_model.h"

typedef struct nn_optim_cls_ADAM_intern
{
    int nbr_layers;
    mat *m_w;
    vec *m_b;
    mat *v_w;
    vec *v_b;
    payload pyl_0, pyl_1;
    size_t t;
    FLT_TYP beta1t, beta2t;
} nn_optim_cls_ADAM_intern;

static nn_optim_cls_ADAM_intern *intern_construct(nn_optim_cls_ADAM_intern *intern, nn_optim *optimizer, const nn_model *model)
{
    assert(intern);
    assert(model);
    assert(optimizer);

    intern->nbr_layers = model->nbr_layers;
    intern->m_w = (mat *)calloc(intern->nbr_layers, sizeof(mat));
    assert(intern->m_w);
    intern->v_w = (mat *)calloc(intern->nbr_layers, sizeof(mat));
    assert(intern->v_w);
    intern->m_b = (vec *)calloc(intern->nbr_layers, sizeof(vec));
    assert(intern->m_b);
    intern->v_b = (vec *)calloc(intern->nbr_layers, sizeof(vec));
    assert(intern->v_b);

    IND_TYP max_sz = 0;

    for (int l = 0; l < intern->nbr_layers; l++)
    {
        IND_TYP inp_sz = (l > 0) ? model->layer[l - 1].out_sz : model->input_size;
        IND_TYP out_sz = model->layer[l].out_sz;
        if (out_sz * inp_sz > max_sz)
            max_sz = out_sz * inp_sz;
        mat_construct(intern->m_w + l, out_sz, inp_sz);
        mat_fill_zero(intern->m_w + l);
        mat_construct(intern->v_w + l, out_sz, inp_sz);
        mat_fill_zero(intern->v_w + l);
        vec_construct(intern->m_b + l, out_sz);
        vec_fill_zero(intern->m_b + l);
        vec_construct(intern->v_b + l, out_sz);
        vec_fill_zero(intern->v_b + l);
    }
    payload_construct(&intern->pyl_0, max_sz);
    assert(payload_is_valid(&intern->pyl_0));
    payload_construct(&intern->pyl_1, max_sz);
    assert(payload_is_valid(&intern->pyl_1));
    nn_optim_cls_ADAM_params *params = (nn_optim_cls_ADAM_params *)optimizer->params;
    intern->t = params->t0;
    intern->beta1t = (FLT_TYP)pow(params->beta1, intern->t);
    intern->beta2t = (FLT_TYP)pow(params->beta2, intern->t);
    return intern;
}

static void intern_destruct(nn_optim_cls_ADAM_intern *intern)
{
    assert(intern);
    for (int l = 0; l < intern->nbr_layers; l++)
    {
        mat_destruct(intern->m_w + l);
        mat_destruct(intern->v_w + l);
        vec_destruct(intern->m_b + l);
        vec_destruct(intern->v_b + l);
    }
    free(intern->m_w);
    free(intern->m_b);
    free(intern->v_w);
    free(intern->v_b);
    payload_release(&intern->pyl_0);
    payload_release(&intern->pyl_1);
    memset(intern, 0, sizeof(nn_optim_cls_ADAM_intern));
}

static nn_optim *nn_optim_cls_ADAM_construct(nn_optim *optimizer, const nn_model *model)
{
    assert(optimizer);
    assert(model);
    optimizer->params = calloc(1, sizeof(nn_optim_cls_ADAM_params));
    assert(optimizer->params);
    optimizer->intern = calloc(1, sizeof(nn_optim_cls_ADAM_intern));
    assert(optimizer->intern);
    nn_optim_cls_ADAM_params *params = (nn_optim_cls_ADAM_params *)optimizer->params;
    nn_optim_cls_ADAM_intern *intern = (nn_optim_cls_ADAM_intern *)optimizer->intern;
    *params = nn_optim_cls_ADAM_params_DEFAULT;
    intern_construct(intern, optimizer, model);

    return optimizer;
}

static void nn_optim_cls_ADAM_destruct(nn_optim *optimizer)
{
    assert(optimizer);
    free(optimizer->params);
    optimizer->params = NULL;
    intern_destruct((nn_optim_cls_ADAM_intern *)optimizer->intern);
    free(optimizer->intern);
    optimizer->intern = NULL;
}

void nn_optim_cls_ADAM_params_clear(nn_optim_cls_ADAM_params *params)
{
    params->alpha = -1;
    params->beta1 = -1;
    params->beta2 = -1;
    params->eps = -1;
    params->t0 = UINT32_MAX;
}

static nn_optim *nn_optim_cls_ADAM_set_params(nn_optim *optimizer, const void *inp_params)
{
    assert(optimizer);
    nn_optim_cls_ADAM_params *params = (nn_optim_cls_ADAM_params *)optimizer->params;
    nn_optim_cls_ADAM_params *inp_p = (nn_optim_cls_ADAM_params *)inp_params;
    if (inp_p)
    {
        if (inp_p->alpha > 0)
            params->alpha = inp_p->alpha;
        if (inp_p->beta1 >= 0 && inp_p->beta1 < 1)
            params->beta1 = inp_p->beta1;
        if (inp_p->beta2 >= 0 && inp_p->beta2 < 1)
            params->beta2 = inp_p->beta2;
        if (inp_p->eps > 0 && inp_p->eps < 1)
            params->eps = inp_p->eps;
        if (inp_p->t0 != UINT32_MAX)
            params->t0 = inp_p->t0;
    }
    return optimizer;
}

static nn_model *nn_optim_cls_ADAM_update_model(nn_optim *optimizer, nn_model *model)
{
    assert(optimizer);
    assert(model);
    nn_optim_cls_ADAM_params *params = (nn_optim_cls_ADAM_params *)optimizer->params;
    nn_optim_cls_ADAM_intern *intern = (nn_optim_cls_ADAM_intern *)optimizer->intern;
    intern->t++;
    FLT_TYP omb1 = 1 - intern->beta1t;
    FLT_TYP omb2 = 1 - intern->beta2t;
    intern->beta1t *= params->beta1;
    intern->beta2t *= params->beta2;
    FLT_TYP c1 = params->beta1 * omb1 / (1 - intern->beta1t);
    FLT_TYP d1 = (1 - params->beta1) / (1 - intern->beta1t);
    FLT_TYP c2 = params->beta2 * omb2 / (1 - intern->beta2t);
    FLT_TYP d2 = (1 - params->beta2) / (1 - intern->beta2t);
    mat tmp_w = mat_NULL;
    mat tmp_dw = mat_NULL;
    vec tmp_b = vec_NULL;
    vec tmp_db = vec_NULL;
    mat_construct_prealloc(&tmp_w, &intern->pyl_0, 0, 1, 1);
    vec_construct_prealloc(&tmp_b, &intern->pyl_1, 0, 1, 1);
    mat_construct_prealloc(&tmp_dw, &intern->pyl_1, 0, 1, 1);
    vec_construct_prealloc(&tmp_db, &intern->pyl_0, 0, 1, 1);
    for (int l = 0; l < model->nbr_layers; l++)
    {
        mat_scale(intern->m_w + l, c1);
        mat_update(intern->m_w + l, d1, model->intern.d_w + l);
        vec_scale(intern->m_b + l, c1);
        vec_update(intern->m_b + l, d1, model->intern.d_b + l);

        mat_reform(&tmp_w, 0, model->intern.d_w[l].d1, model->intern.d_w[l].d2);
        mat_square(&tmp_w, model->intern.d_w + l);
        mat_scale(intern->v_w + l, c2);
        mat_update(intern->v_w + l, d2, &tmp_w);
        vec_reform(&tmp_b, 0, model->intern.d_b[l].d, 1);
        vec_square(&tmp_b, model->intern.d_b + l);
        vec_scale(intern->v_b + l, c2);
        vec_update(intern->v_b + l, d2, &tmp_b);

        mat_sqrt(&tmp_w, intern->v_w + l);
        mat_f_addto(&tmp_w, params->eps);
        mat_reform(&tmp_dw, 0, model->intern.d_w[l].d1, model->intern.d_w[l].d2);
        mat_div(&tmp_dw, intern->m_w + l, &tmp_w);
        mat_update(model->weight + l, -params->alpha, &tmp_dw);

        vec_sqrt(&tmp_b, intern->v_b + l);
        vec_f_addto(&tmp_b, params->eps);
        vec_reform(&tmp_db, 0, model->intern.d_b[l].d, 1);
        vec_div(&tmp_db, intern->m_b + l, &tmp_b);
        vec_update(model->bias + l, -params->alpha, &tmp_db);
    }
    vec_destruct(&tmp_b);
    vec_destruct(&tmp_db);
    mat_destruct(&tmp_w);
    mat_destruct(&tmp_dw);
    return model;
}

const nn_optim_class nn_optim_cls_ADAM = {.construct = nn_optim_cls_ADAM_construct,
                                          .destruct = nn_optim_cls_ADAM_destruct,
                                          .set_params = nn_optim_cls_ADAM_set_params,
                                          .update_model = nn_optim_cls_ADAM_update_model};
