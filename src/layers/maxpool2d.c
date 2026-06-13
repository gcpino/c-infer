/* layers/maxpool2d.c -- NCHW max pooling, stride=K, no padding
 *
 * Params: none. Kernel size K derived from in/out spatial dims:
 *   K = Hin / Hout   (must divide evenly)
 */
#include "cinfer.h"
#include <stdint.h>
#include <math.h>

void cinf_op_maxpool2d(const cinf_tensor_t* in, cinf_tensor_t* out,
                       const float* params, uint32_t params_len) {
    (void)params; (void)params_len;
    const uint32_t C = in->c;
    const uint32_t Hin = in->h, Win = in->w;
    const uint32_t Hout = out->h, Wout = out->w;
    const uint32_t Kh = Hin / Hout;
    const uint32_t Kw = Win / Wout;
    const float* x = in->data;
    float* y = out->data;

    for (uint32_t c = 0; c < C; ++c) {
        for (uint32_t oy = 0; oy < Hout; ++oy) {
            for (uint32_t ox = 0; ox < Wout; ++ox) {
                float m = -INFINITY;
                for (uint32_t ky = 0; ky < Kh; ++ky) {
                    for (uint32_t kx = 0; kx < Kw; ++kx) {
                        float v = x[((c * Hin) + (oy * Kh + ky)) * Win + (ox * Kw + kx)];
                        if (v > m) m = v;
                    }
                }
                y[((c * Hout) + oy) * Wout + ox] = m;
            }
        }
    }
}
