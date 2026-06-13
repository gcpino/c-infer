/* test_sigmoid.c -- Sigmoid parity test. */
#include "cinfer.h"
#include "test_harness.h"
#include <stdlib.h>

int main(void) {
    CINF_RUN(test_sigmoid);
    size_t bsz, esz, isz;
    unsigned char* buf = cinf_read_file("build/test_fixtures/sigmoid.cinf", &bsz);
    unsigned char* exp = cinf_read_file("build/test_fixtures/sigmoid.expected.bin", &esz);
    unsigned char* inb = cinf_read_file("build/test_fixtures/sigmoid.input.bin", &isz);
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
    cinf_free_model(&m);
    free(buf); free(exp); free(inb); free(out);
    return 0;
}
