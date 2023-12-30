#pragma once

#include "nn_config.h"
#include "lin_alg.h"
#include "payload.h"

/**
 * Struct to represent a collection of data points (rows).
 * Each data point is a vector/array of values.
 */
typedef struct data_points
{
    payload payload;    // data payload
    IND_TYP width;      // max row (data point) width
    IND_TYP capacity;   // capacity of nbr_points which can be dynamically changed
    IND_TYP nbr_points; // number of data points stored (nbr_points <= capacity)
    IND_TYP init_capacity;
} data_points;

/**
 * Macro to represent a null/empty data_points object.
 *
 * Sets all fields of the data_points struct to default empty values:
 * - payload: payload_NULL
 * - width: 0
 * - capacity: 0
 * - nbr_points: 0
 */
#define data_points_NULL ((const data_points){.payload = payload_NULL, .width = 0, .capacity = 0, .nbr_points = 0, .init_capacity = 0})

/**
 * The default initial capacity allocated when constructing a new data_points object.
 * This can be overridden in data_points_construct().
 */
#define data_points_DEFAULT_INIT_CAP 4096

/**
 * Constructs a new data_points object with the given width and initial capacity.
 *
 * Allocates memory for the new data_points object and its internal payload.
 * Sets the width, capacity, and initializes nbr_points to 0.
 *
 * @param dtpts Pointer to the data_points object to initialize.
 * @param width The width (number of elements) for each data point.
 * @param init_capacity The initial capacity which is the number of data points that can be stored.
 * @return Pointer to the initialized data_points object.
 */
data_points *data_points_construct(data_points *dtpts, IND_TYP width, IND_TYP init_capacity);

/**
 * Frees the memory allocated for the given data_points object.
 *
 * This releases the memory allocated for the data_points object's
 * internal payload. It should be called when the object is no longer
 * needed.
 */
void data_points_destruct(data_points *dtpts);

// data_points *data_points_replace(data_points *dest, const data_points *src);
// data_points *data_points_insert(data_points *dest, const data_points *src, IND_TYP i);

/**
 * Appends the data points from src to dest.
 *
 * This appends the data points in src to the end of the data points in dest.
 * The capacity of dest is increased if needed to fit all the data points.
 *
 * @param dest The destination data_points object to append to.
 * @param src The source data_points object to append from.
 * @return A pointer to dest after appending.
 */
data_points *data_points_append(data_points *dest, const data_points *src);

// data_points *data_points_append_row(data_points *dest, const vec *raw_row);

/**
 * Gets a data point (row) at the given index from the data points object.
 *
 * @param dtpts The data points object.
 * @param data The output vector to store the retrieved data point.
 * @param i The index of the data point to retrieve.
 * @param dt_sly The slice indicating which part of the data point to retrieve.
 * @return A pointer to the output data vector.
 */
vec *data_points_at(data_points *dtpts, vec *data, IND_TYP i, const slice *dt_sly);

/**
 * Gets a random data point (row) from the data points object.
 *
 * @param dtpts The data points object.
 * @param data The output vector to store the retrieved data point.
 * @param dt_sly The slice indicating which part of the data point to retrieve.
 * @return A pointer to the output data vector.
 */
vec *data_points_at_rnd(data_points *dtpts, vec *data, const slice *dt_sly);

// vec *data_points_column_at(data_points *dtpts, vec *clm, IND_TYP j);

/**
 * Shuffles the data points in the given data_points object.
 *
 * @param dtpts The data_points object containing the data points to shuffle.
 * @param i_sly The slice indicating which part of the data points to shuffle.
 * @return A pointer to the shuffled data_points object.
 */
data_points *data_points_shuffle(data_points *dtpts, slice i_sly);

/**
 * Checks if the given data_points object is valid.
 *
 * This checks that the data_points object is not null, and that its capacity,
 * number of points, and width fields are valid.
 *
 * @param dtpts The data_points object to check.
 * @return True if the data_points object is valid, false otherwise.
 */
bool data_points_is_valid(const data_points *dtpts);

/**
 * Clears the data points in the given data_points object.
 *
 * This resets the number of points to 0 and resets the capacity to init_capacity;
 * The data_points object itself is not freed.
 *
 * @param dtpts The data_points object to clear.
 * @return A pointer to the cleared data_points object.
 */
data_points *data_points_clear(data_points *dtpts);

static inline FLT_TYP *data_points_ptr_at(data_points *dtpts, IND_TYP i)
{
    assert(i >= 0 && i < dtpts->nbr_points);
    return payload_at(&dtpts->payload, i * dtpts->width);
}
