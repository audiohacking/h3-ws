#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Long non-causal DiT attention (row-major [sequence, heads, 128] BF16)
 * against a double CPU oracle on sampled query rows, for each MPSGraph SDPA
 * query-block setting. Usage: h3_sdpa_split_tests [sequence] [heads]. */

enum { HEAD_DIM = 128, SAMPLED_ROWS = 6 };

static uint32_t random_state = 0x6a09e667u;

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_sdpa_split.c: %s\n", message);
    exit(1);
}

static float random_signed(void) {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    return (float)((random_state >> 8) * (1.0 / 16777216.0) * 2.0 - 1.0);
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void reference_row(const uint16_t *query, const uint16_t *key,
                          const uint16_t *value, uint32_t sequence,
                          uint32_t heads, uint32_t row, uint32_t head,
                          double *scores, double *output) {
    const double scale = 1.0 / sqrt((double)HEAD_DIM);
    const uint16_t *q = query + ((size_t)row * heads + head) * HEAD_DIM;
    double maximum = -INFINITY;
    for (uint32_t key_row = 0; key_row < sequence; key_row++) {
        const uint16_t *k = key + ((size_t)key_row * heads + head) * HEAD_DIM;
        double dot = 0.0;
        for (uint32_t d = 0; d < HEAD_DIM; d++)
            dot += (double)bf16_to_f32(q[d]) * bf16_to_f32(k[d]);
        scores[key_row] = dot * scale;
        if (scores[key_row] > maximum) maximum = scores[key_row];
    }
    double sum = 0.0;
    for (uint32_t key_row = 0; key_row < sequence; key_row++) {
        scores[key_row] = exp(scores[key_row] - maximum);
        sum += scores[key_row];
    }
    for (uint32_t d = 0; d < HEAD_DIM; d++) output[d] = 0.0;
    for (uint32_t key_row = 0; key_row < sequence; key_row++) {
        const uint16_t *v = value + ((size_t)key_row * heads + head) * HEAD_DIM;
        double weight = scores[key_row] / sum;
        for (uint32_t d = 0; d < HEAD_DIM; d++)
            output[d] += weight * bf16_to_f32(v[d]);
    }
}

int main(int argc, char **argv) {
    uint32_t sequence = argc > 1 ? (uint32_t)strtoul(argv[1], NULL, 10) : 16385;
    uint32_t heads = argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 10) : 8;
    size_t count = (size_t)sequence * heads * HEAD_DIM;
    uint16_t *query = malloc(count * sizeof(*query));
    uint16_t *key = malloc(count * sizeof(*key));
    uint16_t *value = malloc(count * sizeof(*value));
    uint16_t *got = malloc(count * sizeof(*got));
    double *scores = malloc(sequence * sizeof(*scores));
    double want[HEAD_DIM];
    if (!query || !key || !value || !got || !scores) die("host allocation failed");
    /* Amplitude 3 gives peaked, DiT-like attention rather than a flat mean. */
    for (size_t i = 0; i < count; i++) {
        query[i] = f32_to_bf16(random_signed() * 3.0f);
        key[i] = f32_to_bf16(random_signed() * 3.0f);
        value[i] = f32_to_bf16(random_signed());
    }
    uint32_t rows[SAMPLED_ROWS] = {
        0, 2047, 2048, sequence / 2, sequence - 4097, sequence - 1
    };
    /* Short sequences: out-of-range (or wrapped) samples use the last row. */
    for (uint32_t r = 0; r < SAMPLED_ROWS; r++)
        if (rows[r] >= sequence) rows[r] = sequence - 1;

    char error[512];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "FAIL tests/test_sdpa_split.c: Metal setup: %s\n", error);
        return 1;
    }
    h3_gpu_tensor *q = h3_gpu_tensor_from_bf16(gpu, query, count);
    h3_gpu_tensor *k = h3_gpu_tensor_from_bf16(gpu, key, count);
    h3_gpu_tensor *v = h3_gpu_tensor_from_bf16(gpu, value, count);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, count);
    if (!q || !k || !v || !output) die("Metal tensor allocation failed");

    /* NULL: production default. "0" disables splitting. */
    const char *blocks[] = {NULL, "2048", "8192", "0"};
    int failed = 0;
    for (size_t mode = 0; mode < sizeof(blocks) / sizeof(blocks[0]); mode++) {
        if (blocks[mode]) setenv("H3_SDPA_MAX_QUERY_ROWS", blocks[mode], 1);
        else unsetenv("H3_SDPA_MAX_QUERY_ROWS");
        if (!h3_gpu_begin(gpu) ||
            !h3_gpu_sdpa_bf16(gpu, output, q, k, v, sequence, heads, HEAD_DIM,
                              1.0f / sqrtf((float)HEAD_DIM)) ||
            !h3_gpu_submit(gpu)) {
            fprintf(stderr, "FAIL tests/test_sdpa_split.c: SDPA: %s\n",
                    h3_gpu_error(gpu));
            return 1;
        }
        if (!h3_gpu_tensor_read_bf16(output, got, count)) die("readback failed");
        double worst = 0.0;
        for (uint32_t r = 0; r < SAMPLED_ROWS; r++)
            for (uint32_t head = 0; head < heads; head += heads > 4 ? heads / 4 : 1) {
                reference_row(query, key, value, sequence, heads, rows[r], head,
                              scores, want);
                const uint16_t *row = got + ((size_t)rows[r] * heads + head) *
                                      HEAD_DIM;
                for (uint32_t d = 0; d < HEAD_DIM; d++) {
                    double error_abs = fabs((double)bf16_to_f32(row[d]) - want[d]);
                    if (!(error_abs <= worst)) worst = error_abs;
                }
            }
        /* BF16 output rounding of values <= 1 is at most ~0.004. */
        int ok = worst < 0.02;
        printf("SDPA S=%u heads=%u block=%-8s: max abs %.4g %s\n", sequence,
               heads, blocks[mode] ? blocks[mode] : "default", worst,
               ok ? "ok" : "MISMATCH");
        /* Only the production default gates the test; the others report. */
        if (!ok && !blocks[mode]) failed = 1;
    }
    /* The video VAE decoder runs the same graph in F32. */
    unsetenv("H3_SDPA_MAX_QUERY_ROWS");
    float *f32_input[3];
    const uint16_t *bf16_input[3] = {query, key, value};
    h3_gpu_tensor *f32_tensor[4];
    for (int t = 0; t < 3; t++) {
        f32_input[t] = malloc(count * sizeof(float));
        if (!f32_input[t]) die("host allocation failed");
        for (size_t i = 0; i < count; i++)
            f32_input[t][i] = bf16_to_f32(bf16_input[t][i]);
        f32_tensor[t] = h3_gpu_tensor_from_f32(gpu, f32_input[t], count);
    }
    f32_tensor[3] = h3_gpu_tensor_new_f32(gpu, count);
    if (!f32_tensor[0] || !f32_tensor[1] || !f32_tensor[2] || !f32_tensor[3])
        die("Metal F32 tensor allocation failed");
    if (!h3_gpu_begin(gpu) ||
        !h3_gpu_sdpa_f32(gpu, f32_tensor[3], f32_tensor[0], f32_tensor[1],
                         f32_tensor[2], sequence, heads, HEAD_DIM,
                         1.0f / sqrtf((float)HEAD_DIM)) ||
        !h3_gpu_submit(gpu)) {
        fprintf(stderr, "FAIL tests/test_sdpa_split.c: F32 SDPA: %s\n",
                h3_gpu_error(gpu));
        return 1;
    }
    float *f32_got = f32_input[0];
    if (!h3_gpu_tensor_read_f32(f32_tensor[3], f32_got, count))
        die("F32 readback failed");
    double f32_worst = 0.0;
    for (uint32_t r = 0; r < SAMPLED_ROWS; r++)
        for (uint32_t head = 0; head < heads; head += heads > 4 ? heads / 4 : 1) {
            reference_row(query, key, value, sequence, heads, rows[r], head,
                          scores, want);
            const float *row = f32_got + ((size_t)rows[r] * heads + head) *
                               HEAD_DIM;
            for (uint32_t d = 0; d < HEAD_DIM; d++) {
                double error_abs = fabs((double)row[d] - want[d]);
                if (!(error_abs <= f32_worst)) f32_worst = error_abs;
            }
        }
    printf("SDPA S=%u heads=%u f32 default : max abs %.4g %s\n", sequence,
           heads, f32_worst, f32_worst < 1e-3 ? "ok" : "MISMATCH");
    if (!(f32_worst < 1e-3)) failed = 1;
    for (int t = 0; t < 4; t++) h3_gpu_tensor_free(f32_tensor[t]);
    for (int t = 0; t < 3; t++) free(f32_input[t]);
    h3_gpu_free(gpu);
    if (failed) die("default SDPA split disagrees with the CPU oracle");
    puts("ok: long non-causal SDPA matches the CPU oracle at the default split");
    return 0;
}
