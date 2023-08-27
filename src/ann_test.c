#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "nn.h"


float rnd(void)
{
    return 2.0f * rand() / (float)RAND_MAX - 1;
}

void gen_data(mat_t *data, mat_t *target)
{
    vec_t inp;
    FLT_TYP a = 2;
    FLT_TYP b = -1;
    FLT_TYP c = 0.3;
    FLT_TYP n = 0.03;
    for (int i = 0; i < data->d1; i++)
    {
        vec_init_prealloc(&inp, data->arr + i * data->d2, data->d2);
        vec_fill_rnd(&inp, rnd);
        target->arr[i] =
            a * (1 + n * rnd()) * inp.arr[0] +
            b * (1 + n * rnd()) * inp.arr[1] +
            c * (1 + n * rnd());
    }
}

int main()
{

    srand(time(NULL));
    int nbr_data = 3000;
    int nbr_feat = 2;
    mat_t *data = mat_new(nbr_data, nbr_feat);
    mat_t *target = mat_new(nbr_data, 1);
    gen_data(data, target);
    mat_t *test_data = mat_new(nbr_data/4, nbr_feat);
    mat_t *test_target = mat_new(nbr_data/4, 1);
    gen_data(test_data, test_target);

    nn_layer_t lay0, lay1, lay2;
    lay0.out_sz = 2;
    lay0.activ = nn_activ_RELU;
    lay1.out_sz = 4;
    lay1.activ = nn_activ_RELU;
    lay2.out_sz = 1;
    lay2.activ = nn_activ_ID;


    nn_model_t model = nn_model_NULL;
    nn_model_construct(&model, 8, nbr_feat);
    nn_model_add(&model, &lay0);
    nn_model_add(&model, &lay1);
    nn_model_add(&model, &lay2);
    nn_model_init_rnd(&model, rnd);

    // nn_model_print(&model);
    float avg_err = nn_model_eval(&model, test_data, test_target, nn_err_MSE);
    printf("Eval avg err before training: %f\n", avg_err);

    nn_model_train(&model, data, target, 32, 3000, 0.001, nn_err_MSE, Regression);

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