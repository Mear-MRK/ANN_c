#ifndef NN_ACTIV_H_INCLUDED
#define NN_ACTIV_H_INCLUDED 1

#include "nn_config.h"
#include "lin_alg.h"

typedef vec *(*nn_activation_func)(vec *result, const vec *s);
typedef vec *(*nn_deriv_activ_func)(vec *result, const vec *s, const vec *act_s);

typedef struct nn_activ
{
    nn_activation_func func;
    nn_deriv_activ_func deriv;

} nn_activ;

#define nn_activ_NULL \
    ((const nn_activ) { .func = NULL, .deriv = NULL })

nn_activ *nn_activ_init(nn_activ *activation,
                          const nn_activation_func act_func, const nn_deriv_activ_func deriv);

enum nn_activ_enum
{
    ACTIV_NON = -1,
    ACTIV_ID,
    ACTIV_SIGMOID,
    ACTIV_TANH,
    ACTIV_RELU,
    ACTIV_UNKNOWN
};

extern const nn_activ nn_activ_ID;
extern const nn_activ nn_activ_SIGMOID;
extern const nn_activ nn_activ_TANH;
extern const nn_activ nn_activ_RELU;

char *nn_activ_to_str(const nn_activ *activ, char *string);

nn_activ nn_activ_from_enum(enum nn_activ_enum a);
enum nn_activ_enum nn_activ_to_enum(const nn_activ *activ);

#endif /* NN_ACTIV_H_INCLUDED */