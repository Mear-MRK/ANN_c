#include "nn_activ.h"

#include <assert.h>
#include <string.h>

nn_activ *nn_activ_init(nn_activ *activation,
                          const nn_activation_func act,
                          const nn_deriv_activ_func drv_act)
{
    assert(act);
    assert(drv_act);
    activation->func = act;
    activation->deriv = drv_act;
    return activation;
}

static inline vec *act_id_f(vec *res, const vec *s)
{
    return vec_assign(res, s);
}
static inline vec *act_id_drv(vec *res, const vec *s, const vec *a)
{
    return vec_fill(res, 1);
}
const nn_activ nn_activ_ID = {.func = act_id_f, .deriv = act_id_drv};

static inline vec *sigmoid_drv(vec *res, const vec *s, const vec *a)
{
    vec_f_sub(res, 1, a);
    return vec_mulby(res, a);
}
const nn_activ nn_activ_SIGMOID = {.func = vec_sigmoid, .deriv = sigmoid_drv};

static inline vec *tanh_drv(vec *res, const vec *s, const vec *a)
{
    vec_mul(res, a, a);
    return vec_f_sub(res, 1, res);
}
const nn_activ nn_activ_TANH = {.func = vec_tanh, .deriv = tanh_drv};

static inline vec *relu_drv(vec *res, const vec *s, const vec *a)
{
    return vec_theta(res, s);
}
const nn_activ nn_activ_RELU = {.func = vec_relu, .deriv = relu_drv};


static const char *nn_activ_STR[] = {"NONE", "IDEN", "SIGM", "TANH", "RELU", "UNKNOWN"};

char *nn_activ_to_str(const nn_activ *activ, char *string)
{
    assert(activ);
    assert(string);
    string[0] = 0;
    strcpy(string, nn_activ_STR[1 + nn_activ_to_enum(activ)]);
    return string;
}

nn_activ nn_activ_from_enum(enum nn_activ_enum a)
{
    switch (a)
    {
    case ACTIV_NON:
        return nn_activ_NULL;
    case ACTIV_ID:
        return nn_activ_ID;
    case ACTIV_SIGMOID:
        return nn_activ_SIGMOID;
    case ACTIV_TANH:
        return nn_activ_TANH;
    case ACTIV_RELU:
        return nn_activ_RELU;
    case ACTIV_UNKNOWN:
        return nn_activ_NULL;
    }
    return nn_activ_NULL;
}

enum nn_activ_enum nn_activ_to_enum(const nn_activ *a)
{
    if (memcmp(a, &nn_activ_NULL, sizeof(nn_activ)) == 0)
        return ACTIV_NON;
    if (memcmp(a, &nn_activ_ID, sizeof(nn_activ)) == 0)
        return ACTIV_ID;
    if (memcmp(a, &nn_activ_SIGMOID, sizeof(nn_activ)) == 0)
        return ACTIV_SIGMOID;
    if (memcmp(a, &nn_activ_TANH, sizeof(nn_activ)) == 0)
        return ACTIV_TANH;
    if (memcmp(a, &nn_activ_RELU, sizeof(nn_activ)) == 0)
        return ACTIV_RELU;
    return ACTIV_UNKNOWN;
}
