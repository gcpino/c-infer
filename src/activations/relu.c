/* activations/relu.c -- max(0, x), elementwise. */
#include "cinfer.h"
#include <stdint.h>

void cinf_op_relu(const cinf_tensor_t* in, cinf_tensor_t* out,
                  const float* params, uint32_t params_len) {
    (void)params; (void)params_len;
    const uint32_t n = cinf_numel(in);
    for (uint32_t i = 0; i < n; ++i) {
        float v = in->data[i];
        out->data[i] = v > 0.0f ? v : 0.0f;
    }
}
