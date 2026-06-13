/* activations/sigmoid.c -- 1 / (1 + exp(-x)), elementwise. */
#include "cinfer.h"
#include <stdint.h>
#include <math.h>

void cinf_op_sigmoid(const cinf_tensor_t* in, cinf_tensor_t* out,
                     const float* params, uint32_t params_len) {
    (void)params; (void)params_len;
    const uint32_t n = cinf_numel(in);
    for (uint32_t i = 0; i < n; ++i) {
        out->data[i] = 1.0f / (1.0f + expf(-in->data[i]));
    }
}
