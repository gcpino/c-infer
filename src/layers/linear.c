/* layers/linear.c -- y = x . W^T + b  (no allocations)
 *
 * Param layout (float32, in order):
 *   [0 .. in_features*out_features)    W, row-major, shape (out, in)
 *   [in*out .. in*out + out)            b, length out_features
 *
 * Input  rank 1: shape (N=1, C=in, 1, 1)   numel = in
 * Output rank 1: shape (N=1, C=out, 1, 1)  numel = out
 */
#include "cinfer.h"
#include <string.h>

void cinf_op_linear(const cinf_tensor_t* in, cinf_tensor_t* out,
                    const float* params, uint32_t params_len) {
    (void)params_len;
    const uint32_t in_f  = in->c;
    const uint32_t out_f = out->c;
    const float* W = params;
    const float* b = params + (size_t)in_f * out_f;
    const float* x = in->data;
    float* y = out->data;

    for (uint32_t i = 0; i < out_f; ++i) {
        float acc = b[i];
        const float* wrow = W + (size_t)i * in_f;
        for (uint32_t j = 0; j < in_f; ++j) {
            acc += x[j] * wrow[j];
        }
        y[i] = acc;
    }
}
