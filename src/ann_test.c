#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

#include "nn.h"

float flt_rnd(void)
{
    return rand() / ((float)RAND_MAX + 1.0f);
}

float fill_rnd(void)
{
    return 10 * (flt_rnd() - 0.5);
}

unsigned u_rnd(void)
{
    return (unsigned)rand();
}

void gen_reg_data(mat_t *x, mat_t *trg)
{
    mat_fill_rnd(x, fill_rnd);
    FLT_TYP a0 = 2, a1 = -3;
    FLT_TYP b0 = -1, b1 = 2;
    FLT_TYP c0 = 0.3, c1 = 10.1;
    FLT_TYP n = 0.1;
    for (int i = 0; i < x->d1; i++)
    {
        *mat_at(trg, i, 0) = a0 * (1 + n * (2 * flt_rnd() - 1)) * (*mat_at(x, i, 0)) +
                                b0 * (1 + n * (2 * flt_rnd() - 1)) * (*mat_at(x, i, 1)) +
                                c0 * (1 + n * (2 * flt_rnd() - 1));
        *mat_at(trg, i, 1) = a1 * (1 + n * (2 * flt_rnd() - 1)) * (*mat_at(x, i, 0)) +
                                b1 * (1 + n * (2 * flt_rnd() - 1)) * (*mat_at(x, i, 1)) +
                                c1 * (1 + n * (2 * flt_rnd() - 1));
    }
}

void gen_cat_data(mat_t *x, mat_t *label)
{
    mat_fill_rnd(x, fill_rnd);
    mat_fill_zero(label);
    float noise = 0.1;
    for(int i = 0; i < x->d1; i++)
    {

        FLT_TYP p_x = *mat_at(x, i, 0);
        FLT_TYP p_y = *mat_at(x, i, 1);
        if ( p_x * p_x + 2 * p_y * p_y < 9)
        {
            *mat_at(label, i, 0) = 1;
        } else {
            *mat_at(label, i, 1) = 1;
        }
        if (flt_rnd() < noise)
        {
            FLT_TYP tmp = *mat_at(label, i, 0);
            *mat_at(label, i, 0) = *mat_at(label, i, 1);
            *mat_at(label, i, 1) = tmp;
        }
    }
}

int main()
{
    srand(time(NULL));

    puts("REGRESSION");

    int nbr_data = 3000;
    int nbr_feat = 2;
    int nbr_out = 2;
    mat_t *reg_x = mat_new(nbr_data, nbr_feat);
    mat_t *reg_trg = mat_new(nbr_data, nbr_out);
    gen_reg_data(reg_x, reg_trg);
    mat_t *reg_test_x = mat_new(nbr_data / 4, nbr_feat);
    mat_t *reg_test_trg = mat_new(nbr_data / 4, nbr_out);
    gen_reg_data(reg_test_x, reg_test_trg);

    nn_layer_t lay0, lay1, lay2;
    lay0.out_sz = 4;
    lay0.dropout = 0.1;
    lay0.activ = nn_activ_RELU;
    lay1.out_sz = 8;
    lay1.dropout = 0.1;
    lay1.activ = nn_activ_RELU;

    lay2.out_sz = nbr_out;
    lay2.dropout = 0;
    lay2.activ = nn_activ_ID;


    nn_model_t reg_model = nn_model_NULL;
    nn_model_construct(&reg_model, 8, nbr_feat);
    nn_model_set_rnd_gens(&reg_model, u_rnd, flt_rnd);
    nn_model_add(&reg_model, &lay0);
    nn_model_add(&reg_model, &lay1);
    nn_model_add(&reg_model, &lay2);
    nn_model_init_uniform_rnd(&reg_model, 1, 0);

    nn_optim_t reg_opt;
    // nn_optim_construct(&optimizer, &nn_optim_cls_SGD, &reg_model);
    // nn_optim_cls_SGD_params_t sgd_p;
    // sgd_p.learning_rate = 0.0001f;
    // nn_optim_set_params(&optimizer, &sgd_p);
    nn_optim_construct(&reg_opt, &nn_optim_cls_ADAM, &reg_model);

    // puts("init model:");
    // nn_model_print(&reg_model);
    float avg_err = nn_model_eval(&reg_model, reg_test_x, reg_test_trg, nn_loss_MSE, false);
    printf("Eval avg err before training: %f\n", avg_err);
    puts("Training...");
    nn_model_train(&reg_model, reg_x, reg_trg, 16, 1000, true, &reg_opt, nn_loss_MSE);
    // puts("trained model:");
    // nn_model_print(&reg_model);
    avg_err = nn_model_eval(&reg_model, reg_test_x, reg_test_trg, nn_loss_MSE, false);
    printf("Eval avg err after training: %f\n", avg_err);

    nn_optim_destruct(&reg_opt);
    nn_model_destruct(&reg_model);
    mat_del(reg_test_x);
    mat_del(reg_test_trg);
    mat_del(reg_trg);
    mat_del(reg_x);

    puts("---------------------");
    puts("CATEGORIZATION");

    nbr_data = 10000;
    nbr_feat = 2;
    int nbr_lbl = 2;
    mat_t *cat_x = mat_new(nbr_data, nbr_feat);
    mat_t *cat_lbl = mat_new(nbr_data, nbr_lbl);
    gen_cat_data(cat_x, cat_lbl);
    mat_t *cat_test_x = mat_new(nbr_data / 4, nbr_feat);
    mat_t *cat_test_lbl = mat_new(nbr_data / 4, nbr_lbl);
    gen_cat_data(cat_test_x, cat_test_lbl);

    nn_layer_t cat_lay0, cat_lay1, cat_lay2, cat_lay_end;
    cat_lay0.out_sz = 64;
    cat_lay0.dropout = 0.1;
    cat_lay0.activ = nn_activ_RELU;

    cat_lay_end.out_sz = nbr_lbl;
    cat_lay_end.dropout = 0;
    cat_lay_end.activ = nn_activ_ID;


    nn_model_t cat_model = nn_model_NULL;
    nn_model_construct(&cat_model, 8, nbr_feat);
    nn_model_set_rnd_gens(&cat_model, u_rnd, flt_rnd);
    nn_model_add(&cat_model, &cat_lay0);
    nn_model_add(&cat_model, &cat_lay_end);
    nn_model_init_uniform_rnd(&cat_model, 0.5, 0.01);

    nn_optim_t cat_opt;
    nn_optim_construct(&cat_opt, &nn_optim_cls_ADAM, &cat_model);

    // puts("init model:");
    // nn_model_print(&cat_model);
    avg_err = nn_model_eval(&cat_model, cat_test_x, cat_test_lbl, nn_loss_CrossEnt, true);
    printf("Eval avg err before training: %f\n", avg_err);
    puts("Training...");
    nn_model_train(&cat_model, cat_x, cat_lbl, 16, 1000, true, &cat_opt, nn_loss_CrossEnt);
    // puts("trained model:");
    // nn_model_print(&cat_model);
    avg_err = nn_model_eval(&cat_model, cat_test_x, cat_test_lbl, nn_loss_CrossEnt, true);
    printf("Eval avg err after training: %f\n", avg_err);

    nn_optim_destruct(&cat_opt);
    nn_model_destruct(&cat_model);
    mat_del(cat_test_x);
    mat_del(cat_test_lbl);
    mat_del(cat_lbl);
    mat_del(cat_x);

    return 0;
}