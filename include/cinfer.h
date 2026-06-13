/* cinfer.h -- public API for the c-infer inference engine
 *
 * A pure C11 neural-network inference engine. The engine performs all
 * allocations during cinf_load_model via a bump arena. cinf_forward
 * performs zero dynamic allocations and is safe to call repeatedly.
 *
 * Tensors use row-major contiguous storage. Conv layers assume NCHW.
 * Supported layer/activation kinds are listed in cinf_op_kind_t.
 */
#ifndef CINF_H
#define CINF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CINF_OK = 0,
    CINF_ERR_INVALID_ARG,
    CINF_ERR_OOM,
    CINF_ERR_BAD_MAGIC,
    CINF_ERR_UNSUPPORTED_VERSION,
    CINF_ERR_TRUNCATED,
    CINF_ERR_UNSUPPORTED_OP,
    CINF_ERR_SHAPE,
    CINF_ERR_ENDIAN
} cinf_status_t;

typedef enum {
    CINF_OP_LINEAR   = 1,
    CINF_OP_CONV2D   = 2,
    CINF_OP_MAXPOOL2D= 3,
    CINF_OP_FLATTEN  = 4,
    CINF_OP_RELU     = 5,
    CINF_OP_SIGMOID  = 6,
    CINF_OP_SOFTMAX  = 7
} cinf_op_kind_t;

#define CINF_MAX_RANK 4

typedef struct {
    uint8_t* base;
    size_t   capacity;
    size_t   offset;
} cinf_arena_t;

typedef struct {
    uint32_t rank;
    uint32_t n, c, h, w;
    float*   data;
} cinf_tensor_t;

/* Internal: a single op node in the static execution graph. */
typedef struct cinf_op_node cinf_op_node_t;

typedef struct {
    uint32_t         n_layers;
    cinf_op_node_t*  layers;
    cinf_arena_t     arena;
    /* First activation is the model input. */
    cinf_tensor_t*   input;
    /* Last activation is the model output. */
    cinf_tensor_t*   output;
} cinf_model_t;

/* Parse a .cinf blob and lay out a model in a freshly-internal arena.
 * On CINF_OK, *out owns a contiguous arena that can be released with
 * cinf_free_model. On any error, *out is in a defined "empty" state. */
cinf_status_t cinf_load_model(const void* buffer, size_t size, cinf_model_t* out);

/* Release all resources owned by a successfully-loaded model. */
void cinf_free_model(cinf_model_t* model);

/* Run inference. Performs no dynamic allocations. */
cinf_status_t cinf_forward(const cinf_model_t* model,
                           const float* input, float* output);

/* Human-readable string for a status code. Never returns NULL. */
const char* cinf_strerror(cinf_status_t status);

#ifdef __cplusplus
}
#endif

/* === internal helpers shared across translation units =================
 * Not part of the public API; visible so the engine and per-layer
 * object files can call them without a separate internal header. */
void* cinf_arena_alloc(cinf_arena_t* a, size_t n, size_t align);
uint32_t cinf_numel(const cinf_tensor_t* t);
int      cinf_shape_eq(const cinf_tensor_t* a, const cinf_tensor_t* b);
void     cinf_shape_copy(cinf_tensor_t* dst, const cinf_tensor_t* src);
size_t   cinf_shape_total(const cinf_tensor_t* t);

/* Per-op forward declarations, called by cinf_forward. */
void cinf_op_linear   (const cinf_tensor_t* in, cinf_tensor_t* out, const float* params, uint32_t params_len);
void cinf_op_conv2d   (const cinf_tensor_t* in, cinf_tensor_t* out, const float* params, uint32_t params_len);
void cinf_op_maxpool2d(const cinf_tensor_t* in, cinf_tensor_t* out, const float* params, uint32_t params_len);
void cinf_op_flatten  (const cinf_tensor_t* in, cinf_tensor_t* out, const float* params, uint32_t params_len);
void cinf_op_relu     (const cinf_tensor_t* in, cinf_tensor_t* out, const float* params, uint32_t params_len);
void cinf_op_sigmoid  (const cinf_tensor_t* in, cinf_tensor_t* out, const float* params, uint32_t params_len);
void cinf_op_softmax  (const cinf_tensor_t* in, cinf_tensor_t* out, const float* params, uint32_t params_len);

#endif /* CINF_H */
