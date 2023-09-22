#ifndef NN_OPTIM_H_INCLUDED
#define NN_OPTIM_H_INCLUDED 1

#include "nn_model.h"

struct nn_optim_struct;
typedef struct nn_optim_struct nn_optim_t;

typedef nn_model_t *(*nn_optim_update_func)(nn_optim_t *optimizer, nn_model_t *model);
typedef void (*nn_optim_construct_func)(nn_optim_t *optimizer, const nn_model_t* model, FLT_TYP optim_rate);
typedef void (*nn_optim_destruct_func)(nn_optim_t *optimizer);

struct nn_optim_struct
{
    nn_optim_update_func step;
    nn_optim_construct_func construct;
    nn_optim_destruct_func destruct;
    FLT_TYP optim_rate;
    int nbr_m_states;
    mat_t* m_intern_state;
    int nbr_v_states;
    vec_t* v_intern_state;

};

#define nn_optim_NULL \
    (nn_optim_t) { .step = NULL, .construct = NULL, .destruct = NULL, .optim_rate = 0, .nbr_m_states = 0, .m_intern_state = NULL, .nbr_v_states = 0, .v_intern_state = NULL }

nn_optim_t* nn_optim_construct(nn_optim_t *optimizer, const nn_model_t* model, FLT_TYP optim_rate);
void nn_optim_destruct(nn_optim_t *optimizer);


extern nn_optim_t nn_optim_SGD;

#endif /* NN_OPTIM_H_INCLUDED */