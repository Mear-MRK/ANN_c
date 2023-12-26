#include "data_points.h"

#include <string.h>
#include <assert.h>

#include "rnd.h"
#include "log.h"

data_points_t *data_points_construct(data_points_t *dtpts, IND_TYP width, IND_TYP init_capacity)
{
    assert(dtpts);
    assert(width > 0);
    assert(init_capacity >= 0);

    if (!dtpts)
    {
        log_msg(LOG_ERR, "data_points_construct: dtpts is NULL!");
        return NULL;
    }
    if (width <= 0 || init_capacity < 0)
    {
        log_msg(LOG_ERR, "data_points_construct: cannot construct the dtpts with these params!");
        *dtpts = data_points_NULL;
        return dtpts;
    }
    dtpts->init_capacity = (init_capacity) ? init_capacity : data_points_DEFAULT_INIT_CAP;
    dtpts->capacity = dtpts->init_capacity;
    dtpts->width = width;
    dtpts->payload = payload_NULL;
    payload_construct(&dtpts->payload, dtpts->width * dtpts->capacity);
    assert(payload_is_valid(&dtpts->payload));
    if (!payload_is_valid(&dtpts->payload))
    {
        log_msg(LOG_ERR, "data_points_construct: cannot construct the payload!");
        *dtpts = data_points_NULL;
    }

    return dtpts;
}

void data_points_destruct(data_points_t *dtpts)
{
    assert(data_points_is_valid(dtpts));
    if (!dtpts)
        return;
    payload_release(&dtpts->payload);
    *dtpts = data_points_NULL;
}

data_points_t *data_points_append(data_points_t *dest, const data_points_t *src)
{
    assert(data_points_is_valid(dest));
    assert(data_points_is_valid(src));

    IND_TYP new_nbr = dest->nbr_points + src->nbr_points;
    if (dest->capacity < new_nbr)
    {
        IND_TYP new_cap = 3 * new_nbr / 2;
        if (payload_resize(&dest->payload, new_cap))
            dest->capacity = new_cap;
        else
            log_msg(LOG_WRN, "data_points_append: cannot resize the payload!");
    }
    dest->nbr_points += payload_copy(&dest->payload, dest->nbr_points, &src->payload, 0, 0);
    return dest;
}

vec_t *data_points_at(data_points_t *dtpts, vec_t *data, IND_TYP i, const slice_t *sly)
{
    assert(data_points_is_valid(dtpts));
    assert(data);

    if (i < 0)
        i += dtpts->nbr_points;
    assert(i < dtpts->nbr_points && i >= 0);

    if (!sly || slice_is_none(sly))
        return vec_construct_prealloc(data, &dtpts->payload, i * dtpts->width, dtpts->width, 1);

    assert(slice_is_valid(sly));
    assert(slice_is_regulated(sly));

    return vec_construct_prealloc(data, &dtpts->payload, i * dtpts->width + sly->start, sly->len, sly->step);
}

vec_t *data_points_at_rnd(data_points_t *dtpts, vec_t *data, const slice_t *sly)
{
    assert(data_points_is_valid(dtpts));
    assert(vec_is_valid(data));
    IND_TYP i = (IND_TYP)(UINT_RND_GEN() % dtpts->nbr_points);
    return data_points_at(dtpts, data, i, sly);
}

data_points_t *data_points_shuffle(data_points_t *dtpts, slice_t i_sly)
{
    assert(data_points_is_valid(dtpts));
    assert(slice_is_valid(&i_sly));

    slice_regulate(&i_sly, dtpts->nbr_points);

    vec_t tmp = vec_NULL, va = vec_NULL, vb = vec_NULL;
    vec_construct(&tmp, dtpts->width);
    for (IND_TYP k = 0; k < i_sly.len - 1; k++)
    {
        IND_TYP l = k + (IND_TYP)(UINT_RND_GEN() % (i_sly.len - k));
        if (l != k)
        {
            data_points_at(dtpts, &va, slice_index(&i_sly, k), NULL);
            data_points_at(dtpts, &vb, slice_index(&i_sly, l), NULL);
            vec_assign(&tmp, &va);
            vec_assign(&va, &vb);
            vec_assign(&vb, &tmp);
        }
    }
    vec_destruct(&tmp);
    vec_destruct(&va);
    vec_destruct(&vb);
    return dtpts;
}

static bool is_null(const data_points_t *dtpts)
{
    assert(dtpts);
    return memcmp(dtpts, &data_points_NULL, sizeof(data_points_t)) == 0;
}

bool data_points_is_valid(const data_points_t *dtpts)
{
    return dtpts &&
           !is_null(dtpts) &&
           dtpts->capacity > 0 &&
           dtpts->width > 0 &&
           dtpts->nbr_points >= 0 &&
           dtpts->nbr_points <= dtpts->capacity &&
           payload_is_valid(&dtpts->payload) &&
           (size_t)(dtpts->capacity * dtpts->width) <= dtpts->payload.size;
}

data_points_t *data_points_clear(data_points_t *dtpts)
{
    assert(data_points_is_valid(dtpts));
    dtpts->nbr_points = 0;
    if (!payload_resize(&dtpts->payload, dtpts->init_capacity))
    {
        log_msg(LOG_WRN, "data_points_clear: cannot resize the payload!");
    }
    return dtpts;
}
