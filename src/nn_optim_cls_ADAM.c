#include "nn_optim_cls_ADAM.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "nn_model.h"

typedef struct
{
    int nbr_layers;
    mat_t *m_w;
    vec_t *m_b;
    mat_t *v_w;
    vec_t *v_b;
    FLT_TYP *tmp0, *tmp1;
    size_t t;
    FLT_TYP beta1t, beta2t;

} nn_optim_cls_ADAM_intern_t;

static nn_optim_cls_ADAM_intern_t *intern_construct(nn_optim_cls_ADAM_intern_t *intern, nn_optim_t *optimizer, const nn_model_t *model)
{
    assert(intern);
    intern->nbr_layers = model->nbr_layers;
    intern->m_w = (mat_t *)calloc(intern->nbr_layers, sizeof(mat_t));
    assert(intern->m_w);
    intern->v_w = (mat_t *)calloc(intern->nbr_layers, sizeof(mat_t));
    assert(intern->v_w);
    intern->m_b = (vec_t *)calloc(intern->nbr_layers, sizeof(vec_t));
    assert(intern->m_b);
    intern->v_b = (vec_t *)calloc(intern->nbr_layers, sizeof(vec_t));
    assert(intern->v_b);

    IND_TYP max_sz = 0;

    for (int l = 0; l < intern->nbr_layers; l++)
    {
        IND_TYP inp_sz = (l != 0) ? model->layer[l - 1].out_sz : model->input_size;
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
    intern->tmp0 = (FLT_TYP *)calloc(max_sz, sizeof(FLT_TYP));
    assert(intern->tmp0);
    intern->tmp1 = (FLT_TYP *)calloc(max_sz, sizeof(FLT_TYP));
    assert(intern->tmp1);
    nn_optim_cls_ADAM_params_t *params = (nn_optim_cls_ADAM_params_t *)optimizer->params;
    intern->t = params->t0;
    intern->beta1t = (FLT_TYP)pow(params->beta1, intern->t);
    intern->beta2t = (FLT_TYP)pow(params->beta2, intern->t);
    return intern;
}

static void intern_destruct(nn_optim_cls_ADAM_intern_t *intern)
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
    free(intern->tmp0);
    free(intern->tmp1);
    memset(intern, 0, sizeof(nn_optim_cls_ADAM_intern_t));
}

static nn_optim_t *nn_optim_cls_ADAM_construct(nn_optim_t *optimizer, const nn_model_t *model)
{
    assert(optimizer);
    assert(model);
    optimizer->params = calloc(1, sizeof(nn_optim_cls_ADAM_params_t));
    optimizer->intern = calloc(1, sizeof(nn_optim_cls_ADAM_intern_t));
    assert(optimizer->params);
    nn_optim_cls_ADAM_params_t *params = (nn_optim_cls_ADAM_params_t *)optimizer->params;
    nn_optim_cls_ADAM_intern_t *intern = (nn_optim_cls_ADAM_intern_t *)optimizer->intern;
    *params = nn_optim_cls_ADAM_params_DEFAULT;
    intern_construct(intern, optimizer, model);

    return optimizer;
}

static void nn_optim_cls_ADAM_destruct(nn_optim_t *optimizer)
{
    assert(optimizer);
    free(optimizer->params);
    optimizer->params = NULL;
    intern_destruct((nn_optim_cls_ADAM_intern_t *)optimizer->intern);
    optimizer->intern = NULL;
}

void nn_optim_cls_ADAM_params_clear(nn_optim_cls_ADAM_params_t *params)
{
    params->alpha = -1;
    params->beta1 = -1;
    params->beta2 = -1;
    params->eps = -1;
    params->t0 = UINT32_MAX;
}

static nn_optim_t *nn_optim_cls_ADAM_set_params(nn_optim_t *optimizer, const void *inp_params)
{
    assert(optimizer);
    nn_optim_cls_ADAM_params_t *params = (nn_optim_cls_ADAM_params_t *)optimizer->params;
    nn_optim_cls_ADAM_params_t *inp_p = (nn_optim_cls_ADAM_params_t *)inp_params;
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

static nn_model_t *nn_optim_cls_ADAM_update_model(nn_optim_t *optimizer, nn_model_t *model)
{
    assert(optimizer);
    assert(model);
    nn_optim_cls_ADAM_params_t *params = (nn_optim_cls_ADAM_params_t *)optimizer->params;
    nn_optim_cls_ADAM_intern_t *intern = (nn_optim_cls_ADAM_intern_t *)optimizer->intern;
    intern->t++;
    FLT_TYP omb1 = 1 - intern->beta1t;
    FLT_TYP omb2 = 1 - intern->beta2t;
    intern->beta1t *= params->beta1;
    intern->beta2t *= params->beta2;
    FLT_TYP c1 = params->beta1 * omb1 / (1 - intern->beta1t);
    FLT_TYP d1 = (1 - params->beta1) / (1 - intern->beta1t);
    FLT_TYP c2 = params->beta2 * omb2 / (1 - intern->beta2t);
    FLT_TYP d2 = (1 - params->beta2) / (1 - intern->beta2t);
    mat_t tmp_w;
    mat_t tmp_dw;
    vec_t tmp_b;
    vec_t tmp_db;
    for (int l = 0; l < model->nbr_layers; l++)
    {
        mat_scale(intern->m_w + l, c1);
        mat_update(intern->m_w + l, d1, model->intern.d_w + l);
        vec_scale(intern->m_b + l, c1);
        vec_update(intern->m_b + l, d1, model->intern.d_b + l);

        mat_init_prealloc(&tmp_w, intern->tmp0, model->intern.d_w[l].d1, model->intern.d_w[l].d2);
        mat_square(&tmp_w, model->intern.d_w + l);
        mat_scale(intern->v_w + l, c2);
        mat_update(intern->v_w + l, d2, &tmp_w);
        vec_init_prealloc(&tmp_b, intern->tmp1, model->intern.d_b[l].size);
        vec_square(&tmp_b, model->intern.d_b + l);
        vec_scale(intern->v_b + l, c2);
        vec_update(intern->v_b + l, d2, &tmp_b);

        mat_sqrt(&tmp_w, intern->v_w + l);
        mat_f_addto(&tmp_w, params->eps);
        mat_init_prealloc(&tmp_dw, intern->tmp1, model->intern.d_w[l].d1, model->intern.d_w[l].d2);
        mat_div(&tmp_dw, intern->m_w + l, &tmp_w);
        mat_update(model->weight + l, -params->alpha, &tmp_dw);

        vec_sqrt(&tmp_b, intern->v_b + l);
        vec_f_addto(&tmp_b, params->eps);
        vec_init_prealloc(&tmp_db, intern->tmp0, model->intern.d_b[l].size);
        vec_div(&tmp_db, intern->m_b + l, &tmp_b);
        vec_update(model->bias + l, -params->alpha, &tmp_db);
    }

    return model;
}

const nn_optim_class nn_optim_cls_ADAM = {.construct = nn_optim_cls_ADAM_construct,
                                          .destruct = nn_optim_cls_ADAM_destruct,
                                          .set_params = nn_optim_cls_ADAM_set_params,
                                          .update_model = nn_optim_cls_ADAM_update_model};
