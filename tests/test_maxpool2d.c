/* test_maxpool2d.c -- MaxPool2D parity test (exact). */
#include "cinfer.h"
#include "test_harness.h"
#include <stdlib.h>

int main(void) {
    CINF_RUN(test_maxpool2d);
    size_t bsz, esz, isz;
    unsigned char* buf = cinf_read_file("build/test_fixtures/maxpool2d.cinf", &bsz);
    unsigned char* exp = cinf_read_file("build/test_fixtures/maxpool2d.expected.bin", &esz);
    unsigned char* inb = cinf_read_file("build/test_fixtures/maxpool2d.input.bin", &isz);
    if (!buf || !exp || !inb) CINF_FAIL("missing fixture");

    cinf_model_t m;
    CINF_ASSERT_STATUS(cinf_load_model(buf, bsz, &m), CINF_OK);

    uint32_t out_n = (uint32_t)(esz / sizeof(float));
    float* out = (float*)calloc(out_n, sizeof(float));
    CINF_ASSERT_STATUS(cinf_forward(&m, (const float*)inb, out), CINF_OK);

    const float* expected = (const float*)exp;
    /* Max pooling is exact (selects max of a small discrete set). */
    for (uint32_t i = 0; i < out_n; ++i) {
        CINF_ASSERT_EQ(*(uint32_t*)&out[i], *(uint32_t*)&expected[i]);
    }

    cinf_free_model(&m);
    free(buf); free(exp); free(inb); free(out);
    return 0;
}
