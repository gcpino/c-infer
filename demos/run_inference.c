/* run_inference.c -- load a .cinf file and run inference on a fixed
 * demo input, printing the result. Used by `make demo`.
 *
 *   ./build/run_inference build/mlp.cinf [input.bin]
 *
 * If no input file is given, reads `build/mlp.input.bin` next to the
 * .cinf file (matching the dump from mlp_demo.py).
 */
#include "cinfer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int load_file(const char* path, unsigned char** out, size_t* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char* b = (unsigned char*)malloc((size_t)sz);
    if (fread(b, 1, (size_t)sz, f) != (size_t)sz) {
        fprintf(stderr, "short read on %s\n", path);
        free(b); fclose(f); return -1;
    }
    fclose(f);
    *out = b; *out_size = (size_t)sz;
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <file.cinf> [input.bin]\n", argv[0]);
        return 2;
    }
    const char* cinf_path = argv[1];
    unsigned char* blob; size_t bsz;
    if (load_file(cinf_path, &blob, &bsz) != 0) return 1;

    cinf_model_t m;
    cinf_status_t s = cinf_load_model(blob, bsz, &m);
    if (s != CINF_OK) {
        fprintf(stderr, "cinf_load_model: %s\n", cinf_strerror(s));
        free(blob); return 1;
    }
    printf("loaded: %u layers, in=(%u,%u,%u,%u) out=(%u,%u,%u,%u)\n",
           m.n_layers,
           m.input->n, m.input->c, m.input->h, m.input->w,
           m.output->n, m.output->c, m.output->h, m.output->w);

    /* Load input. */
    const char* in_path = (argc >= 3) ? argv[2] : "build/mlp.input.bin";
    unsigned char* inb; size_t isz;
    if (load_file(in_path, &inb, &isz) != 0) { cinf_free_model(&m); free(blob); return 1; }
    uint32_t in_n = cinf_numel(m.input);
    if ((uint32_t)(isz / sizeof(float)) != in_n) {
        fprintf(stderr, "input size mismatch: model wants %u floats, file has %zu\n",
                in_n, isz / sizeof(float));
        cinf_free_model(&m); free(blob); free(inb); return 1;
    }

    uint32_t out_n = cinf_numel(m.output);
    float* out = (float*)calloc(out_n, sizeof(float));

    s = cinf_forward(&m, (const float*)inb, out);
    if (s != CINF_OK) {
        fprintf(stderr, "cinf_forward: %s\n", cinf_strerror(s));
        cinf_free_model(&m); free(blob); free(inb); free(out); return 1;
    }

    printf("C output (%u): [", out_n);
    for (uint32_t i = 0; i < out_n; ++i) {
        printf("%s%.6f", (i ? ", " : ""), (double)out[i]);
    }
    printf("]\n");
    float sum = 0.0f;
    for (uint32_t i = 0; i < out_n; ++i) sum += out[i];
    printf("sum: %.6f\n", (double)sum);

    cinf_free_model(&m);
    free(blob); free(inb); free(out);
    return 0;
}
