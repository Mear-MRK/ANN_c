#include "nn_activ.h"

#include <assert.h>

nn_activ_t *nn_activ_init(nn_activ_t *activation,
 const nn_activation_func act,
 const nn_deriv_activ_func drv_act)
{
    assert(act);
    assert(drv_act);
    activation->func = act;
    activation->deriv = drv_act;
    return activation;
}