/* test_harness.h -- minimal plain-C test harness.
 *
 *   CINF_ASSERT_EQ(actual, expected)         integer/uint32 equality
 *   CINF_ASSERT_NEAR(actual, expected, tol)  float32 absolute diff
 *   CINF_ASSERT_STATUS(call, expected)       cinf_status_t check
 *
 * The user writes a main() that runs assertions and returns 0 on
 * success or 1 on first failure (via CINF_FAIL).
 */
#ifndef CINF_TEST_HARNESS_H
#define CINF_TEST_HARNESS_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int cinf_test_failed = 0;
static const char* cinf_test_name = "<unnamed>";

#define CINF_RUN(name) do { cinf_test_name = #name; printf("==> %s\n", #name); } while (0)

#define CINF_FAIL(msg) do { \
    fprintf(stderr, "FAIL [%s] %s:%d: %s\n", cinf_test_name, __FILE__, __LINE__, msg); \
    cinf_test_failed = 1; return 1; \
} while (0)

#define CINF_ASSERT_EQ(actual, expected) do { \
    long long _a = (long long)(actual), _e = (long long)(expected); \
    if (_a != _e) { \
        fprintf(stderr, "FAIL [%s] %s:%d: expected %lld, got %lld\n", \
                cinf_test_name, __FILE__, __LINE__, _e, _a); \
        cinf_test_failed = 1; return 1; \
    } \
} while (0)

#define CINF_ASSERT_NEAR(actual, expected, tol) do { \
    double _a = (double)(actual), _e = (double)(expected); \
    double _d = _a - _e; if (_d < 0) _d = -_d; \
    if (_d > (double)(tol)) { \
        fprintf(stderr, "FAIL [%s] %s:%d: |%g - %g| = %g > %g\n", \
                cinf_test_name, __FILE__, __LINE__, _a, _e, _d, (double)(tol)); \
        cinf_test_failed = 1; return 1; \
    } \
} while (0)

#define CINF_ASSERT_STATUS(call, expected_status) do { \
    cinf_status_t _s = (call); \
    if (_s != (expected_status)) { \
        fprintf(stderr, "FAIL [%s] %s:%d: expected status %d (%s), got %d (%s)\n", \
                cinf_test_name, __FILE__, __LINE__, \
                (int)(expected_status), cinf_strerror(expected_status), \
                (int)_s, cinf_strerror(_s)); \
        cinf_test_failed = 1; return 1; \
    } \
} while (0)

/* Read the whole contents of a file into a freshly-malloced buffer.
 * On success, *out_size is set and *out_buf must be free()'d by the
 * caller. On failure, returns NULL. */
static unsigned char* cinf_read_file(const char* path, size_t* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char* buf = (unsigned char*)malloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        fprintf(stderr, "short read on %s\n", path);
        free(buf); fclose(f); return NULL;
    }
    fclose(f);
    *out_size = (size_t)sz;
    return buf;
}

#endif /* CINF_TEST_HARNESS_H */
