/* layers/conv2d.c -- single-batch 2D convolution, NCHW, stride=1, padding=0
 *
 * Param layout (float32, in order):
 *   [0 .. out_c*in_c*K*K)   W, shape (out_c, in_c, K, K) row-major
 *   [out*in*K*K .. +out_c)  b, length out_c
 *
 * Kernel size K is derived from the input/output spatial dims:
 *   K = Hin - Hout + 1   (== Win - Wout + 1; spec mandates square)
 */
#include "cinfer.h"
#include <stdint.h>

void cinf_op_conv2d(const cinf_tensor_t* in, cinf_tensor_t* out,
                    const float* params, uint32_t params_len) {
    (void)params_len;
    const uint32_t in_c  = in->c;
    const uint32_t out_c = out->c;
    const uint32_t Hin = in->h, Win = in->w;
    const uint32_t Hout = out->h, Wout = out->w;
    const uint32_t K = Hin - Hout + 1;

    const float* W = params;
    const float* b = params + (size_t)out_c * in_c * K * K;
    const float* x = in->data;
    float* y = out->data;

    for (uint32_t oc = 0; oc < out_c; ++oc) {
        for (uint32_t oy = 0; oy < Hout; ++oy) {
            for (uint32_t ox = 0; ox < Wout; ++ox) {
                float acc = b[oc];
                for (uint32_t ic = 0; ic < in_c; ++ic) {
                    for (uint32_t ky = 0; ky < K; ++ky) {
                        for (uint32_t kx = 0; kx < K; ++kx) {
                            float w = W[((oc * in_c + ic) * K + ky) * K + kx];
                            float xv = x[((ic * Hin) + (oy + ky)) * Win + (ox + kx)];
                            acc += w * xv;
                        }
                    }
                }
                y[((oc * Hout) + oy) * Wout + ox] = acc;
            }
        }
    }
}
