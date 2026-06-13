/* test_linear.c -- Linear parity test (vs PyTorch-generated reference). */
#include "cinfer.h"
#include "test_harness.h"
#include <string.h>

int main(void) {
    CINF_RUN(test_linear);

    size_t bsz, esz, isz;
    unsigned char* buf = cinf_read_file("build/test_fixtures/linear.cinf", &bsz);
    if (!buf) CINF_FAIL("read .cinf");
    unsigned char* exp = cinf_read_file("build/test_fixtures/linear.expected.bin", &esz);
    if (!exp) { free(buf); CINF_FAIL("read expected"); }
    unsigned char* inb = cinf_read_file("build/test_fixtures/linear.input.bin", &isz);
    if (!inb) { free(buf); free(exp); CINF_FAIL("read input"); }

    cinf_model_t m;
    CINF_ASSERT_STATUS(cinf_load_model(buf, bsz, &m), CINF_OK);

    /* Output buffer is sized by the model's output shape. */
    uint32_t out_n = (uint32_t)(esz / sizeof(float));
    float* out = (float*)calloc(out_n, sizeof(float));
    CINF_ASSERT_STATUS(cinf_forward(&m, (const float*)inb, out), CINF_OK);

    const float* expected = (const float*)exp;
    for (uint32_t i = 0; i < out_n; ++i) {
        CINF_ASSERT_NEAR(out[i], expected[i], 1e-5);
    }

    cinf_free_model(&m);
    free(buf); free(exp); free(inb); free(out);
    return 0;
}
