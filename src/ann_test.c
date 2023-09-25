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

void gen_data(mat_t *data, mat_t *target)
{
    mat_fill_rnd(data, fill_rnd);
    FLT_TYP a0 = 2, a1 = -3;
    FLT_TYP b0 = -1, b1 = 2;
    FLT_TYP c0 = 0.3, c1 = 10.1;
    FLT_TYP n = 0.1;
    for (int i = 0; i < data->d1; i++)
    {
        *mat_at(target, i, 0) = a0 * (1 + n * (2 * flt_rnd() - 1)) * (*mat_at(data, i, 0)) +
                                b0 * (1 + n * (2 * flt_rnd() - 1)) * (*mat_at(data, i, 1)) +
                                c0 * (1 + n * (2 * flt_rnd() - 1));
        *mat_at(target, i, 1) = a1 * (1 + n * (2 * flt_rnd() - 1)) * (*mat_at(data, i, 0)) +
                                b1 * (1 + n * (2 * flt_rnd() - 1)) * (*mat_at(data, i, 1)) +
                                c1 * (1 + n * (2 * flt_rnd() - 1));
    }
}

int main()
{
    srand(time(NULL));
    int nbr_data = 3000;
    int nbr_feat = 2;
    int nbr_out = 2;
    mat_t *data = mat_new(nbr_data, nbr_feat);
    mat_t *target = mat_new(nbr_data, nbr_out);
    gen_data(data, target);
    mat_t *test_data = mat_new(nbr_data / 4, nbr_feat);
    mat_t *test_target = mat_new(nbr_data / 4, nbr_out);
    gen_data(test_data, test_target);

    nn_layer_t lay0, lay1, lay2;
    lay0.out_sz = 4;
    lay0.dropout = 0.01;
    lay0.activ = nn_activ_RELU;
    lay1.out_sz = 8;
    lay1.dropout = 0.04;
    lay1.activ = nn_activ_RELU;
    lay2.out_sz = nbr_out;
    lay2.dropout = 0.02;
    lay2.activ = nn_activ_ID;

    nn_model_t model = nn_model_NULL;
    nn_model_construct(&model, 8, nbr_feat);
    nn_model_set_rnd_gens(&model, u_rnd, flt_rnd);
    nn_model_add(&model, &lay0);
    nn_model_add(&model, &lay1);
    nn_model_add(&model, &lay2);
    nn_model_init_rnd(&model, 1, 0);

    puts("init model:");
    nn_model_print(&model);
    float avg_err = nn_model_eval(&model, test_data, test_target, nn_err_MSE);
    printf("Eval avg err before training: %f\n", avg_err);
    puts("Training...");
    nn_model_train(&model, data, target, 16, 10000, true, 0.0001f, nn_err_MSE);
    puts("trained model:");
    nn_model_print(&model);
    avg_err = nn_model_eval(&model, test_data, test_target, nn_err_MSE);
    printf("Eval avg err after training: %f\n", avg_err);
    nn_model_destruct(&model);

    mat_del(test_data);
    mat_del(test_target);
    mat_del(target);
    mat_del(data);

    return 0;
}