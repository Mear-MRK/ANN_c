#ifndef NN_LOSS_H_INCLUDED
#define NN_LOSS_H_INCLUDED 1

#include "nn_config.h"
#include "lin_alg.h"

typedef FLT_TYP (*nn_loss_func)(const vec *target, const vec *output, vec *buff);
typedef vec *(*nn_deriv_loss_func)(vec *result, const vec *target, const vec *output);

typedef struct nn_loss
{
    nn_loss_func func;
    nn_deriv_loss_func deriv;

} nn_loss;

#define nn_loss_NULL ((const nn_loss){.func = NULL, .deriv = NULL})

nn_loss *nn_loss_init(nn_loss *loss, const nn_loss_func err_func, const nn_deriv_loss_func deriv);

// mean square err
extern const nn_loss nn_loss_MSE;
// categorical cross entropy
extern const nn_loss nn_loss_CrossEnt;

enum nn_loss_enum
{
    LOSS_NON = -1,
    LOSS_MSE,   // mean square err
    LOSS_CCE,   // categorical cross entropy
    LOSS_UNKNOWN
};

nn_loss nn_loss_from_enum(const enum nn_loss_enum e);
enum nn_loss_enum nn_loss_to_enum(const nn_loss *loss);


#endif /* NN_LOSS_H_INCLUDED */