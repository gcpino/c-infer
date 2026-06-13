/* arena.c -- bump allocator used during cinf_load_model only. */
#include "cinfer.h"
#include <stdint.h>
#include <string.h>

/* Internal helper: reserve n bytes from the arena with given alignment.
 * Returns NULL on overflow. Used by engine.c and the layer modules. */
void* cinf_arena_alloc(cinf_arena_t* a, size_t n, size_t align) {
    if (a == NULL || a->base == NULL) return NULL;
    if (align == 0) align = 1;
    /* align offset up */
    size_t aligned = (a->offset + (align - 1)) & ~(align - 1);
    if (aligned > a->capacity) return NULL;
    if (n > a->capacity - aligned) return NULL;
    void* p = a->base + aligned;
    a->offset = aligned + n;
    return p;
}
