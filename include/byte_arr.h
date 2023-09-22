#pragma once

#include <stdlib.h>

typedef struct byte_arr_struct
{
    char *arr;
    size_t sz;
} byte_arr_t;

#define byte_arr_NULL ((byte_arr_t){.arr = NULL, .sz = 0});

byte_arr_t *byte_arr_construct(byte_arr_t* ba, size_t size);
void byte_arr_destruct(byte_arr_t* ba);

byte_arr_t *byte_arr_new(size_t size);
void byte_arr_del(byte_arr_t* ba);