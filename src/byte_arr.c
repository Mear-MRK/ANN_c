#include "byte_arr.h"

#include <assert.h>

byte_arr_t *byte_arr_construct(byte_arr_t *ba, size_t size)
{
    if (size == 0) {
        *ba = byte_arr_NULL;
        return ba;
    }
    ba->arr = (char*) calloc(size, 1);
    assert(ba->arr);
    ba->sz= size;
    return ba;
}

void byte_arr_destruct(byte_arr_t *ba)
{
    assert(ba);
    if(ba->arr)
        free(ba->arr);
    *ba = byte_arr_NULL;
}

byte_arr_t *byte_arr_new(size_t size)
{
    byte_arr_t* ba = (byte_arr_t*) malloc(sizeof(byte_arr_t));
    assert(ba);
    return byte_arr_construct(ba, size);
}

void byte_arr_del(byte_arr_t *ba)
{
    assert(ba);
    byte_arr_destruct(ba);
    free(ba);
}
