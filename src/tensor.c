/* tensor.c -- shape helpers and a non-allocating arena constructor. */
#include "cinfer.h"
#include <string.h>

uint32_t cinf_numel(const cinf_tensor_t* t) {
    if (t == NULL) return 0;
    switch (t->rank) {
        case 1: return t->n;
        case 2: return t->n * t->c;
        case 4: return t->n * t->c * t->h * t->w;
        default: return 0;
    }
}

int cinf_shape_eq(const cinf_tensor_t* a, const cinf_tensor_t* b) {
    if (a == NULL || b == NULL) return 0;
    if (a->rank != b->rank) return 0;
    if (a->n != b->n) return 0;
    if (a->c != b->c) return 0;
    if (a->h != b->h) return 0;
    if (a->w != b->w) return 0;
    return 1;
}

/* Copy shape fields, expanding the smaller rank with 1s. */
void cinf_shape_copy(cinf_tensor_t* dst, const cinf_tensor_t* src) {
    if (dst == NULL || src == NULL) return;
    dst->rank = src->rank;
    dst->n = src->n; dst->c = src->c; dst->h = src->h; dst->w = src->w;
}

size_t cinf_shape_total(const cinf_tensor_t* t) {
    return (size_t)cinf_numel(t);
}
