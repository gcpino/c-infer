/* activations/softmax.c -- numerically stable softmax.
 *
 * Normalizes across the last non-trivial dim of the input:
 *   (1, 5, 1, 1) -> across c=5
 *   (1, 2, 8, 8) -> across W=8 (per (n, c, h) row)
 *   (n)          -> whole vector
 *
 * "Last non-trivial dim" = the last dim with size > 1.
 *
 * Uses the max-subtract trick for numerical stability.
 */
#include "cinfer.h"
#include <stdint.h>
#include <math.h>

void cinf_op_softmax(const cinf_tensor_t* in, cinf_tensor_t* out,
                     const float* params, uint32_t params_len) {
    (void)params; (void)params_len;
    const uint32_t n = cinf_numel(in);
    if (n == 0) return;

    uint32_t shape[4] = {in->n, in->c, in->h, in->w};
    uint32_t inner = 1;
    for (int d = 3; d >= 0; --d) {
        if (shape[d] > 1) { inner = shape[d]; break; }
    }
    if (inner == 1) inner = n;
    uint32_t outer = n / inner;

    const float* x = in->data;
    float* y = out->data;

    for (uint32_t o = 0; o < outer; ++o) {
        const float* xr = x + (size_t)o * inner;
        float* yr = y + (size_t)o * inner;

        float m = xr[0];
        for (uint32_t i = 1; i < inner; ++i) {
            if (xr[i] > m) m = xr[i];
        }
        float sum = 0.0f;
        for (uint32_t i = 0; i < inner; ++i) {
            float e = expf(xr[i] - m);
            yr[i] = e;
            sum += e;
        }
        float inv = 1.0f / sum;
        for (uint32_t i = 0; i < inner; ++i) yr[i] *= inv;
    }
}
