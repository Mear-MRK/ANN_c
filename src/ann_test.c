#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>

#include "nn.h"
#include "log.h"

static inline FLT_TYP flt_rnd(void)
{
    return rand() / ((FLT_TYP)RAND_MAX + 1);
}

static inline FLT_TYP fill_rnd(void)
{
    return 10 * (flt_rnd() - 0.5);
}

static inline unsigned u_rnd(void)
{
    return (unsigned)rand();
}

void gen_reg_data(data_points *x, data_points *trg)
{
    x->nbr_points = x->capacity;
    trg->nbr_points = trg->capacity;
    assert(data_points_is_valid(x));
    assert(data_points_is_valid(trg));
    assert(x->capacity == trg->capacity);

    mat x_m = mat_NULL;
    mat_construct_prealloc(&x_m, &x->payload, 0, x->capacity, x->width);
    mat_fill_rnd(&x_m, fill_rnd);
    FLT_TYP a0 = 2, a1 = -3;
    FLT_TYP b0 = -1, b1 = 2;
    FLT_TYP c0 = 5, c1 = -8;
    FLT_TYP n = 0.1;
    mat trg_m = mat_NULL;
    mat_construct_prealloc(&trg_m, &trg->payload, 0, trg->capacity, trg->width);
    for (int i = 0; i < x->nbr_points; i++)
    {
        FLT_TYP px, py;
        px = *mat_at(&x_m, i, 0);
        py = *mat_at(&x_m, i, 1);
        *mat_at(&trg_m, i, 0) = (1 + n * (2 * flt_rnd() - 1)) * px * px +
                                a0 * (1 + n * (2 * flt_rnd() - 1)) * py * py +
                                b0 * (1 + n * (2 * flt_rnd() - 1)) * px * py +
                                c0 * (1 + n * (2 * flt_rnd() - 1)) * (px + py);
        *mat_at(&trg_m, i, 1) = -(1 + n * (2 * flt_rnd() - 1)) * px * px +
                                a1 * (1 + n * (2 * flt_rnd() - 1)) * py * py +
                                b1 * (1 + n * (2 * flt_rnd() - 1)) * px * py +
                                c1 * (1 + n * (2 * flt_rnd() - 1)) * (px - py);
    }
    // char bfs[8096];
    // printf("trg: %s\n", mat_to_str(&trg_m, bfs));
    mat_destruct(&x_m);
    mat_destruct(&trg_m);
}

void gen_cat_data(data_points *x, data_points *trg)
{
    x->nbr_points = x->capacity;
    trg->nbr_points = trg->capacity;
    assert(data_points_is_valid(x));
    assert(data_points_is_valid(trg));
    assert(x->capacity == trg->capacity);

    mat x_m = mat_NULL;
    mat lbl_m = mat_NULL;
    mat_construct_prealloc(&x_m, &x->payload, 0, x->capacity, x->width);
    mat_construct_prealloc(&lbl_m, &trg->payload, 0, trg->capacity, trg->width);

    mat_fill_rnd(&x_m, fill_rnd);
    mat_fill_zero(&lbl_m);
    FLT_TYP noise = 0.1;
    for (int i = 0; i < x_m.d1; i++)
    {

        FLT_TYP p_x = *mat_at(&x_m, i, 0);
        FLT_TYP p_y = *mat_at(&x_m, i, 1);
        if (p_x * p_x + 2 * p_y * p_y < 25)
        {
            *mat_at(&lbl_m, i, 0) = 1;
        }
        else
        {
            *mat_at(&lbl_m, i, 1) = 1;
        }
        if (flt_rnd() < noise)
        {
            FLT_TYP tmp = *mat_at(&lbl_m, i, 0);
            *mat_at(&lbl_m, i, 0) = *mat_at(&lbl_m, i, 1);
            *mat_at(&lbl_m, i, 1) = tmp;
        }
    }
    mat_destruct(&x_m);
    mat_destruct(&lbl_m);
}

int main()
{
    srand(time(NULL));
#ifdef DEBUG
    log_set_level(LOG_INF);
#endif

    int nbr_data = 6000;
    FLT_TYP test_ratio = 0.2;
    int nbr_feat = 2;
    int nbr_ep = 300;
    int batch_sz = 16;

    puts("--------------");
    puts("REGRESSi0N");
    puts("--------------");
    int nbr_out = 2;

    data_points reg_x, reg_trg;
    data_points_construct(&reg_x, nbr_feat, nbr_data);
    data_points_construct(&reg_trg, nbr_out, nbr_data);
    gen_reg_data(&reg_x, &reg_trg);
    slice reg_dt_sly, reg_tst_sly;
    IND_TYP dt_end = round((1 - test_ratio) * nbr_data);
    slice_set(&reg_dt_sly, 0, dt_end, 1);
    slice_set(&reg_tst_sly, dt_end, slice_IND_P_INF, 1);

    nn_layer reg_lay0, reg_lay1, reg_lay_end;
    reg_lay0.out_sz = 8;
    reg_lay0.dropout = 0.1;
    reg_lay0.activ = nn_activ_TANH;
    reg_lay1.out_sz = 16;
    reg_lay1.dropout = 0.1;
    reg_lay1.activ = nn_activ_RELU;

    reg_lay_end.out_sz = nbr_out;
    reg_lay_end.dropout = 0;
    reg_lay_end.activ = nn_activ_ID;

    nn_model reg_model = nn_model_NULL;
    nn_model_construct(&reg_model, 8, nbr_feat);
    nn_model_append(&reg_model, &reg_lay0);
    nn_model_append(&reg_model, &reg_lay1);
    nn_model_append(&reg_model, &reg_lay_end);
    nn_model_init_uniform_rnd(&reg_model, 0.1, 0);

    nn_optim reg_opt;
    // nn_optim_construct(&optimizer, &nn_optim_cls_SGD, &reg_model);
    // nn_optim_cls_SGD_params sgd_p;
    // sgd_p.learning_rate = 0.0001f;
    // nn_optim_set_params(&optimizer, &sgd_p);
    nn_optim_construct(&reg_opt, &nn_optim_cls_ADAM, &reg_model);

    // puts("init model:");
    // nn_model_print(&reg_model);
    printf("Model nbr of parameters: %lu\n", nn_model_nbr_param(&reg_model));
    FLT_TYP avg_err = nn_model_eval(&reg_model, &reg_x, slice_NONE, &reg_trg, slice_NONE, NULL, reg_tst_sly, nn_loss_MSE, false);
    printf("Eval avg err before training: %f\n", avg_err);
    nn_model_train(&reg_model, &reg_x, slice_NONE, &reg_trg, slice_NONE, NULL, reg_dt_sly, batch_sz, nbr_ep, true, &reg_opt, nn_loss_MSE);
    // puts("trained model:");
    // nn_model_print(&reg_model);
    avg_err = nn_model_eval(&reg_model, &reg_x, slice_NONE, &reg_trg, slice_NONE, NULL, reg_tst_sly, nn_loss_MSE, false);
    printf("Eval avg err after training: %f\n", avg_err);

    nn_optim_destruct(&reg_opt);
    nn_model_destruct(&reg_model);
    data_points_destruct(&reg_x);
    data_points_destruct(&reg_trg);

    puts("--------------");
    puts("ClASSifiCATi0N");
    puts("--------------");
    int nbr_lbl = 2;

    data_points cat_x, cat_trg;
    data_points_construct(&cat_x, nbr_feat, nbr_data);
    data_points_construct(&cat_trg, nbr_out, nbr_data);
    gen_cat_data(&cat_x, &cat_trg);
    slice cat_dt_sly, cat_tst_sly;
    dt_end = round((1 - test_ratio) * nbr_data);
    slice_set(&cat_dt_sly, 0, dt_end, 1);
    slice_set(&cat_tst_sly, dt_end, slice_IND_P_INF, 1);

    nn_layer cat_lay0, cat_lay_end;
    cat_lay0.out_sz = 64;
    cat_lay0.dropout = 0.1;
    cat_lay0.activ = nn_activ_RELU;

    cat_lay_end.out_sz = nbr_lbl;
    cat_lay_end.dropout = 0;
    cat_lay_end.activ = nn_activ_ID;

    nn_model cat_model = nn_model_NULL;
    nn_model_construct(&cat_model, 8, nbr_feat);
    nn_model_append(&cat_model, &cat_lay0);
    nn_model_append(&cat_model, &cat_lay_end);
    nn_model_init_uniform_rnd(&cat_model, 0.5, 0.01);

    nn_optim cat_opt;
    nn_optim_construct(&cat_opt, &nn_optim_cls_ADAM, &cat_model);

    // puts("init model:");
    // nn_model_print(&cat_model);
    printf("Model nbr of parameters: %lu\n", nn_model_nbr_param(&cat_model));
    avg_err = nn_model_eval(&cat_model, &cat_x, slice_NONE, &cat_trg, slice_NONE, NULL, cat_tst_sly, nn_loss_CrossEnt, true);
    printf("Eval inaccuracy before training: %f\n", avg_err);
    nn_model_train(&cat_model, &cat_x, slice_NONE, &cat_trg, slice_NONE, NULL, cat_dt_sly, batch_sz, nbr_ep, true, &cat_opt, nn_loss_CrossEnt);
    // puts("trained model:");
    // nn_model_print(&cat_model);
    avg_err = nn_model_eval(&cat_model, &cat_x, slice_NONE, &cat_trg, slice_NONE, NULL, cat_tst_sly, nn_loss_CrossEnt, true);
    printf("Eval inaccuracy after training: %f\n", avg_err);

    nn_optim_destruct(&cat_opt);
    nn_model_destruct(&cat_model);
    data_points_destruct(&cat_x);
    data_points_destruct(&cat_trg);
    return 0;
}