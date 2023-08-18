#ifndef NN_ACTIV_H_INCLUDED
#define NN_ACTIV_H_INCLUDED 1

#include "lin_alg.h"

typedef vec_t *(*nn_activation_func)(vec_t *result, const vec_t *v);
typedef vec_t *(*nn_deriv_activ_func)(vec_t *result, const vec_t *v, const vec_t *act_v);

typedef struct nn_activ_struct
{
    nn_activation_func func;
    nn_deriv_activ_func deriv;

} nn_activ_t;

#define nn_activ_NULL \
    (nn_activ_t) { .func = NULL, .deriv = NULL }

nn_activ_t *nn_activ_init(nn_activ_t *activation, 
const nn_activation_func act_func, const nn_deriv_activ_func deriv);

#endif /* NN_ACTIV_H_INCLUDED */