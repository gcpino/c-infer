/* test_softmax.c -- Softmax parity test. */
#include "cinfer.h"
#include "test_harness.h"
#include <stdlib.h>

int main(void) {
    CINF_RUN(test_softmax);
    size_t bsz, esz, isz;
    unsigned char* buf = cinf_read_file("build/test_fixtures/softmax.cinf", &bsz);
    unsigned char* exp = cinf_read_file("build/test_fixtures/softmax.expected.bin", &esz);
    unsigned char* inb = cinf_read_file("build/test_fixtures/softmax.input.bin", &isz);
    if (!buf || !exp || !inb) CINF_FAIL("missing fixture");

    cinf_model_t m;
    CINF_ASSERT_STATUS(cinf_load_model(buf, bsz, &m), CINF_OK);

    uint32_t out_n = (uint32_t)(esz / sizeof(float));
    float* out = (float*)calloc(out_n, sizeof(float));
    CINF_ASSERT_STATUS(cinf_forward(&m, (const float*)inb, out), CINF_OK);

    const float* expected = (const float*)exp;
    for (uint32_t i = 0; i < out_n; ++i) {
        CINF_ASSERT_NEAR(out[i], expected[i], 1e-6);
    }
    /* Sanity: sums to 1 within 1e-5. */
    float s = 0.0f;
    for (uint32_t i = 0; i < out_n; ++i) s += out[i];
    CINF_ASSERT_NEAR(s, 1.0f, 1e-5);
    cinf_free_model(&m);
    free(buf); free(exp); free(inb); free(out);
    return 0;
}
