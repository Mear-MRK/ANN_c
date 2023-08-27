#ifndef NN_ERR_H_INCLUDED
#define NN_ERR_H_INCLUDED 1

#include "lin_alg.h"

typedef FLT_TYP (*nn_err_func)(const vec_t *target, const vec_t *output, vec_t *buff);
typedef vec_t *(*nn_deriv_err_func)(vec_t *result, const vec_t *target, const vec_t *output);

typedef struct nn_err_struct
{
    nn_err_func func;
    nn_deriv_err_func deriv;

} nn_err_t;

#define nn_err_NULL \
    (nn_err_t) { .func = NULL, .deriv = NULL }

nn_err_t *nn_err_init(nn_err_t *err,
                      const nn_err_func err_func, const nn_deriv_err_func deriv);


extern const nn_err_t nn_err_MSE;

#endif /* NN_ERR_H_INCLUDED */