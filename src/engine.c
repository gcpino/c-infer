/* engine.c -- .cinf loader, static graph executor, error reporting. */
#include "cinfer.h"
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/* ===== op node definition (internal) =================================== */

struct cinf_op_node {
    cinf_op_kind_t kind;
    cinf_tensor_t* in;   /* points into a separately-allocated array of cinf_tensor_t */
    cinf_tensor_t* out;
    uint32_t       params_len;
    float*         params;  /* points into the arena */
};

/* ===== error string ==================================================== */

const char* cinf_strerror(cinf_status_t s) {
    switch (s) {
        case CINF_OK:                    return "ok";
        case CINF_ERR_INVALID_ARG:       return "invalid argument";
        case CINF_ERR_OOM:               return "out of memory";
        case CINF_ERR_BAD_MAGIC:         return "bad file magic (expected 'CINF')";
        case CINF_ERR_UNSUPPORTED_VERSION:return "unsupported .cinf version";
        case CINF_ERR_TRUNCATED:         return "truncated .cinf file";
        case CINF_ERR_UNSUPPORTED_OP:    return "unsupported op kind in .cinf";
        case CINF_ERR_SHAPE:             return "invalid or inconsistent tensor shape";
        case CINF_ERR_ENDIAN:            return "host is big-endian; c-infer requires little-endian";
        default:                         return "unknown error";
    }
}

/* ===== little-endian readers ========================================== */

static uint32_t rd_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static int host_is_little_endian(void) {
    uint16_t x = 1;
    return *(uint8_t*)&x == 1;
}

/* ===== parameter-size computation per op ============================== */

static cinf_status_t params_size(cinf_op_kind_t k,
                                 const cinf_tensor_t* in,
                                 const cinf_tensor_t* out,
                                 uint32_t* out_params_count) {
    switch (k) {
        case CINF_OP_LINEAR: {
            uint32_t in_f  = in->c;
            uint32_t out_f = out->c;
            *out_params_count = in_f * out_f + out_f;
            return CINF_OK;
        }
        case CINF_OP_CONV2D: {
            if (in->h < out->h || in->w < out->w) return CINF_ERR_SHAPE;
            uint32_t K = in->h - out->h + 1;
            if (in->w - out->w + 1 != K) return CINF_ERR_SHAPE;
            *out_params_count = out->c * in->c * K * K + out->c;
            return CINF_OK;
        }
        case CINF_OP_MAXPOOL2D:
        case CINF_OP_FLATTEN:
        case CINF_OP_RELU:
        case CINF_OP_SIGMOID:
        case CINF_OP_SOFTMAX:
            *out_params_count = 0;
            return CINF_OK;
        default:
            return CINF_ERR_UNSUPPORTED_OP;
    }
}

/* ===== cinf_free_model ================================================ */

void cinf_free_model(cinf_model_t* m) {
    if (m == NULL) return;
    if (m->arena.base != NULL) {
        free(m->arena.base);
        m->arena.base = NULL;
    }
    m->layers = NULL;
    m->n_layers = 0;
    m->input = NULL;
    m->output = NULL;
}

/* ===== parser ========================================================= */

#define CINF_MAGIC_0 'C'
#define CINF_MAGIC_1 'I'
#define CINF_MAGIC_2 'N'
#define CINF_MAGIC_3 'F'
#define CINF_VERSION  1u

/* Header layout (16 bytes, little-endian):
 *   0..3   magic "CINF"
 *   4..7   version (uint32)
 *   8..11  n_layers (uint32)
 *   12..15 weight_payload_bytes (uint32) -- advisory, we re-derive
 *
 * Per-layer record (variable):
 *   uint32  op_kind
 *   uint32  in_rank
 *   uint32  in_shape[4]   (unused dims = 1)
 *   uint32  out_rank
 *   uint32  out_shape[4]
 *   uint32  param_bytes
 *   byte[]  param_payload
 */

static const uint8_t* advance(const uint8_t* p, const uint8_t* end, size_t n) {
    if ((size_t)(end - p) < n) return NULL;
    return p + n;
}

cinf_status_t cinf_load_model(const void* buffer, size_t size, cinf_model_t* m) {
    if (buffer == NULL || m == NULL) return CINF_ERR_INVALID_ARG;
    if (!host_is_little_endian()) return CINF_ERR_ENDIAN;
    memset(m, 0, sizeof(*m));

    const uint8_t* p   = (const uint8_t*)buffer;
    const uint8_t* end = p + size;

    /* --- header --- */
    if ((size_t)(end - p) < 16) return CINF_ERR_TRUNCATED;
    if (p[0] != CINF_MAGIC_0 || p[1] != CINF_MAGIC_1 ||
        p[2] != CINF_MAGIC_2 || p[3] != CINF_MAGIC_3) {
        return CINF_ERR_BAD_MAGIC;
    }
    uint32_t version = rd_u32(p + 4);
    if (version != CINF_VERSION) return CINF_ERR_UNSUPPORTED_VERSION;
    uint32_t n_layers = rd_u32(p + 8);
    /* uint32_t weight_bytes = rd_u32(p + 12); -- advisory, re-derived */
    p += 16;

    /* --- two passes:
     *   pass 1: compute per-layer param sizes and per-tensor output sizes
     *   pass 2: actually lay out and copy
     */

    /* First, allocate the per-layer node array and per-tensor metadata.
     * We need (n_layers + 1) tensor slots: slot 0 is the input, slot
     * i+1 is the output of layer i. The total output buffer size is
     * the sum of (n_layers) layer-output numel, plus the input numel. */

    /* We need to parse all layer records once just to know the shapes.
     * To avoid re-parsing, we save a copy of the parsed intermediate
     * state. But records can be large; simpler: re-walk. We stash the
     * records' "view" in a temp array sized n_layers. */

    typedef struct {
        cinf_op_kind_t kind;
        cinf_tensor_t  in_view;   /* shape only */
        cinf_tensor_t  out_view;  /* shape only */
        uint32_t       param_count;
        const uint8_t* param_ptr; /* points into the input buffer */
    } parsed_t;

    parsed_t* parsed = (parsed_t*)calloc(n_layers, sizeof(parsed_t));
    if (parsed == NULL) { cinf_free_model(m); return CINF_ERR_OOM; }

    /* Pass 1: read records, validate, count elements. */
    {
        cinf_tensor_t prev_out;
        memset(&prev_out, 0, sizeof(prev_out));
        for (uint32_t i = 0; i < n_layers; ++i) {
            parsed_t* r = &parsed[i];
            /* op_kind */
            p = advance(p, end, 4); if (!p) goto trunc;
            r->kind = (cinf_op_kind_t)rd_u32(p - 4);
            /* in_shape: rank + 4 dims */
            p = advance(p, end, 4); if (!p) goto trunc;
            r->in_view.rank = rd_u32(p - 4);
            p = advance(p, end, 16); if (!p) goto trunc;
            r->in_view.n = rd_u32(p - 16);
            r->in_view.c = rd_u32(p - 12);
            r->in_view.h = rd_u32(p - 8);
            r->in_view.w = rd_u32(p - 4);
            /* out_shape */
            p = advance(p, end, 4); if (!p) goto trunc;
            r->out_view.rank = rd_u32(p - 4);
            p = advance(p, end, 16); if (!p) goto trunc;
            r->out_view.n = rd_u32(p - 16);
            r->out_view.c = rd_u32(p - 12);
            r->out_view.h = rd_u32(p - 8);
            r->out_view.w = rd_u32(p - 4);
            /* param_bytes */
            p = advance(p, end, 4); if (!p) goto trunc;
            uint32_t pb = rd_u32(p - 4);
            /* param payload */
            if (pb % 4 != 0) { free(parsed); return CINF_ERR_TRUNCATED; }
            p = advance(p, end, pb); if (!p) goto trunc;
            r->param_ptr  = p - pb;
            r->param_count = pb / 4;

            /* Validate: output of layer i-1 must match input of layer i. */
            if (i == 0) {
                prev_out = r->in_view;
            } else {
                if (!cinf_shape_eq(&prev_out, &r->in_view)) {
                    free(parsed); return CINF_ERR_SHAPE;
                }
            }
            /* Compute and store param count via params_size helper. */
            uint32_t pc = 0;
            cinf_status_t st = params_size(r->kind, &r->in_view, &r->out_view, &pc);
            if (st != CINF_OK) { free(parsed); return st; }
            if (pc != r->param_count) { free(parsed); return CINF_ERR_TRUNCATED; }
            prev_out = r->out_view;
        }
    }

    /* --- Compute arena layout:
     *   [ arena header bookkeeping ]
     *   [ cinf_op_node_t * n_layers        ]
     *   [ cinf_tensor_t  * (n_layers + 1)  ]  -- 0 = model input, i+1 = out of layer i
     *   [ weight/bias float payload        ]
     *   [ activation float buffers         ]  -- one per tensor slot, each sized to its
     *                                            layer's numel
     *
     * Each tensor slot owns its own data buffer. Layers may safely read
     * `in` and write `out` even if the in-place aliasing would corrupt
     * the result (e.g. Linear in_f > out_f). */

    uint32_t total_params = 0;
    uint32_t total_act     = 0;
    for (uint32_t i = 0; i < n_layers; ++i) {
        uint32_t n = cinf_numel(&parsed[i].out_view);
        total_params += parsed[i].param_count;
        total_act    += n;
    }
    uint32_t input_numel = cinf_numel(&parsed[0].in_view);
    total_act += input_numel;

    size_t nodes_bytes   = (size_t)n_layers * sizeof(cinf_op_node_t);
    size_t tensors_bytes = (size_t)(n_layers + 1) * sizeof(cinf_tensor_t);
    size_t params_bytes  = (size_t)total_params * sizeof(float);
    size_t act_bytes     = (size_t)total_act * sizeof(float);

    /* align each segment to 16 bytes for safety */
    size_t off = 0;
    off = (off + 15) & ~(size_t)15;  size_t off_nodes   = off; off += nodes_bytes;
    off = (off + 15) & ~(size_t)15;  size_t off_tensors = off; off += tensors_bytes;
    off = (off + 15) & ~(size_t)15;  size_t off_params  = off; off += params_bytes;
    off = (off + 15) & ~(size_t)15;  size_t off_act     = off; off += act_bytes;

    m->arena.capacity = off;
    m->arena.base = (uint8_t*)malloc(m->arena.capacity);
    if (m->arena.base == NULL) { free(parsed); cinf_free_model(m); return CINF_ERR_OOM; }
    m->arena.offset = 0;
    memset(m->arena.base, 0, m->arena.capacity);

    cinf_op_node_t* nodes = (cinf_op_node_t*)(m->arena.base + off_nodes);
    cinf_tensor_t*  tens  = (cinf_tensor_t* )(m->arena.base + off_tensors);
    float*          sparams = (float*        )(m->arena.base + off_params);
    float*          sact    = (float*        )(m->arena.base + off_act);

    m->n_layers = n_layers;
    m->layers   = nodes;
    m->input    = &tens[0];
    m->output   = &tens[n_layers];

    /* Per-tensor data cursor (each tensor gets its own buffer). */
    float* act_cursor = sact;

    /* Slot 0 is the model input. */
    tens[0] = parsed[0].in_view;
    tens[0].data = act_cursor;
    act_cursor += input_numel;

    /* Per-layer param cursor. */
    float* pcursor = sparams;

    for (uint32_t i = 0; i < n_layers; ++i) {
        nodes[i].kind = parsed[i].kind;
        nodes[i].in   = &tens[i];
        nodes[i].out  = &tens[i + 1];
        nodes[i].params_len = parsed[i].param_count;
        nodes[i].params = (parsed[i].param_count > 0) ? pcursor : NULL;

        cinf_shape_copy(&tens[i + 1], &parsed[i].out_view);
        uint32_t out_numel = cinf_numel(&parsed[i].out_view);
        tens[i + 1].data = act_cursor;
        act_cursor += out_numel;

        if (parsed[i].param_count > 0) {
            memcpy(pcursor, parsed[i].param_ptr, (size_t)parsed[i].param_count * sizeof(float));
            pcursor += parsed[i].param_count;
        }
    }

    free(parsed);
    return CINF_OK;

trunc:
    free(parsed);
    cinf_free_model(m);
    return CINF_ERR_TRUNCATED;
}

/* ===== forward dispatch ============================================== */

static void dispatch(const cinf_op_node_t* n) {
    switch (n->kind) {
        case CINF_OP_LINEAR:    cinf_op_linear   (n->in, n->out, n->params, n->params_len); break;
        case CINF_OP_CONV2D:    cinf_op_conv2d   (n->in, n->out, n->params, n->params_len); break;
        case CINF_OP_MAXPOOL2D: cinf_op_maxpool2d(n->in, n->out, n->params, n->params_len); break;
        case CINF_OP_FLATTEN:   cinf_op_flatten  (n->in, n->out, n->params, n->params_len); break;
        case CINF_OP_RELU:      cinf_op_relu     (n->in, n->out, n->params, n->params_len); break;
        case CINF_OP_SIGMOID:   cinf_op_sigmoid  (n->in, n->out, n->params, n->params_len); break;
        case CINF_OP_SOFTMAX:   cinf_op_softmax  (n->in, n->out, n->params, n->params_len); break;
        default: break;
    }
}

cinf_status_t cinf_forward(const cinf_model_t* m, const float* input, float* output) {
    if (m == NULL || m->layers == NULL || input == NULL || output == NULL)
        return CINF_ERR_INVALID_ARG;

    /* Stage input into the shared activation scratch at slot 0. */
    uint32_t in_n = cinf_numel(m->input);
    for (uint32_t i = 0; i < in_n; ++i) m->input->data[i] = input[i];

    for (uint32_t i = 0; i < m->n_layers; ++i) {
        dispatch(&m->layers[i]);
    }

    uint32_t out_n = cinf_numel(m->output);
    for (uint32_t i = 0; i < out_n; ++i) output[i] = m->output->data[i];

    return CINF_OK;
}
