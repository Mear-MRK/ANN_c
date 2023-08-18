#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "nn.h"

vec_t *act_f(vec_t *res, const vec_t *inp)
{
    vec_assign(res, inp);
    return res;
}

vec_t *act_drv(vec_t *res, const vec_t *inp, const vec_t *a)
{
    vec_fill(res, 1);
    return res;
}

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
    int nbr_data = 1000;
    int nbr_feat = 2;
    mat_t *data = mat_new(nbr_data, nbr_feat);
    mat_t *target = mat_new(nbr_data, 1);
    gen_data(data, target);

    nn_activ_t act;
    act.func = act_f;
    act.deriv = act_drv;
    nn_layer_t lay0, lay1;
    lay0.out_sz = 2;
    lay0.activ = act;
    lay1.out_sz = 1;
    lay1.activ = act;

    nn_model_t model = nn_model_NULL;
    nn_model_construct(&model, 8, nbr_feat);
    // nn_model_add(&model, &lay0);
    // nn_model_add(&model, &lay0);
    nn_model_add(&model, &lay1);
    nn_model_init_rnd(&model, rnd);

    nn_model_print(&model);

    nn_model_train(&model, data, target, 1, 1000, 0.01, Regression);

    nn_model_print(&model);

    gen_data(data, target);
    float avg_err = nn_model_eval(&model, data, target);
    printf("Eval avg err: %f\n", avg_err);

    nn_model_destruct(&model);

    mat_del(target);
    mat_del(data);

    return 0;
}