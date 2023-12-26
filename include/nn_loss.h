#ifndef NN_LOSS_H_INCLUDED
#define NN_LOSS_H_INCLUDED 1

#include "lin_alg.h"
#include "nn_conf.h"

typedef FLT_TYP (*nn_loss_func)(const vec_t *target, const vec_t *output, vec_t *buff);
typedef vec_t *(*nn_deriv_loss_func)(vec_t *result, const vec_t *target, const vec_t *output);

typedef struct nn_loss
{
    nn_loss_func func;
    nn_deriv_loss_func deriv;

} nn_loss_t;

#define nn_loss_NULL ((const nn_loss_t){.func = NULL, .deriv = NULL})

nn_loss_t *nn_loss_init(nn_loss_t *loss, const nn_loss_func err_func, const nn_deriv_loss_func deriv);

// mean square err
extern const nn_loss_t nn_loss_MSE;
// categorical cross entropy
extern const nn_loss_t nn_loss_CrossEnt;

enum nn_loss_enum
{
    LOSS_NON = -1,
    LOSS_MSE,   // mean square err
    LOSS_CCE,   // categorical cross entropy
    LOSS_UNKNOWN
};

nn_loss_t nn_loss_from_enum(const enum nn_loss_enum e);
enum nn_loss_enum nn_loss_to_enum(const nn_loss_t *loss);


#endif /* NN_LOSS_H_INCLUDED */