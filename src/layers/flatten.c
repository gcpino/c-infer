/* layers/flatten.c -- pure shape change. Identity for the data buffer. */
#include "cinfer.h"
#include <stdint.h>

void cinf_op_flatten(const cinf_tensor_t* in, cinf_tensor_t* out,
                     const float* params, uint32_t params_len) {
    (void)params; (void)params_len;
    const uint32_t n = cinf_numel(in);
    const uint32_t m = cinf_numel(out);
    if (n != m) return;
    for (uint32_t i = 0; i < n; ++i) out->data[i] = in->data[i];
}
