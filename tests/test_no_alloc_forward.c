/* test_no_alloc_forward.c -- asserts that cinf_forward performs no
 * dynamic allocations. Implemented by wrapping malloc/calloc/realloc/free
 * with the linker --wrap flag, counting calls, and rejecting the run if
 * any allocation occurs during the forward pass.
 *
 * Build with:  -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=realloc -Wl,--wrap=free
 */
#include "cinfer.h"
#include "test_harness.h"
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>

/* Counters updated by the __wrap_* implementations (see linker --wrap). */
static int cinf_alloc_phase = 0;  /* 0 = setup, 1 = forward, 2 = teardown */
static unsigned long cinf_alloc_count = 0;
static unsigned long cinf_free_count  = 0;
static unsigned long cinf_alloc_bytes = 0;

/* Real implementations provided by the wrapped libc. */
extern void* __real_malloc(size_t);
extern void* __real_calloc(size_t, size_t);
extern void* __real_realloc(void*, size_t);
extern void  __real_free(void*);

void* __wrap_malloc(size_t n) {
    if (cinf_alloc_phase == 1) { cinf_alloc_count++; cinf_alloc_bytes += n; }
    return __real_malloc(n);
}
void* __wrap_calloc(size_t n, size_t m) {
    if (cinf_alloc_phase == 1) { cinf_alloc_count++; cinf_alloc_bytes += n * m; }
    return __real_calloc(n, m);
}
void* __wrap_realloc(void* p, size_t n) {
    if (cinf_alloc_phase == 1) { cinf_alloc_count++; cinf_alloc_bytes += n; }
    return __real_realloc(p, n);
}
void  __wrap_free(void* p) {
    if (cinf_alloc_phase == 1) cinf_free_count++;
    __real_free(p);
}

int main(void) {
    CINF_RUN(test_no_alloc_forward);

    size_t bsz, isz;
    unsigned char* buf = cinf_read_file("build/test_fixtures/linear.cinf", &bsz);
    unsigned char* inb = cinf_read_file("build/test_fixtures/linear.input.bin", &isz);
    if (!buf || !inb) CINF_FAIL("missing fixture");

    cinf_model_t m;
    CINF_ASSERT_STATUS(cinf_load_model(buf, bsz, &m), CINF_OK);

    float in[8];
    for (size_t i = 0; i < sizeof(in)/sizeof(in[0]); ++i)
        in[i] = ((const float*)inb)[i];
    float out[5] = {0};

    cinf_alloc_phase = 1;
    cinf_status_t s = cinf_forward(&m, in, out);
    cinf_alloc_phase = 2;

    CINF_ASSERT_STATUS(s, CINF_OK);
    if (cinf_alloc_count != 0 || cinf_free_count != 0) {
        fprintf(stderr,
                "FAIL: cinf_forward performed %lu allocs (%lu bytes) and %lu frees\n",
                cinf_alloc_count, cinf_alloc_bytes, cinf_free_count);
        cinf_free_model(&m);
        free(buf); free(inb);
        return 1;
    }

    cinf_free_model(&m);
    free(buf); free(inb);
    return 0;
}
