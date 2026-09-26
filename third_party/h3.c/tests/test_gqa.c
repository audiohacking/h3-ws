#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    HEAD_DIM = 128,
    TRIALS = 8
};

typedef struct {
    const char *name;
    unsigned sequence;
    unsigned query_heads;
    unsigned kv_heads;
    unsigned trials;
    unsigned row_stride;  /* check every Nth row (plus the last) vs the CPU */
    const char *tile_keys;  /* NULL: production tile size */
    const char *path;  /* "kernel", "mps", or NULL for production routing */
} gqa_case;

static uint32_t random_state;

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_gqa.c: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) die(message);
}

static void require_gpu(h3_gpu *gpu, int condition, const char *operation) {
    if (condition) return;
    fprintf(stderr, "FAIL tests/test_gqa.c: %s: %s\n", operation,
            h3_gpu_error(gpu));
    exit(1);
}

static uint32_t random_u32(void) {
    uint32_t value = random_state;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    random_state = value;
    return value;
}

static float random_signed(void) {
    return (float)((random_u32() >> 8) *
                   (1.0 / 16777216.0) * 2.0 - 1.0);
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

static void make_inputs(const gqa_case *c, unsigned trial, uint16_t *query,
                        uint16_t *key, uint16_t *value) {
    static const float amplitudes[TRIALS] = {
        0.25f, 0.75f, 1.5f, 3.0f, 0.5f, 1.0f, 2.0f, 6.0f
    };
    float amplitude = amplitudes[trial];
    random_state = 0x9e3779b9u ^ (uint32_t)(trial * 0x85ebca6bu);
    for (size_t index = 0;
         index < (size_t)c->sequence * c->query_heads * HEAD_DIM; index++) {
        query[index] = f32_to_bf16(random_signed() * amplitude);
    }
    for (size_t index = 0;
         index < (size_t)c->sequence * c->kv_heads * HEAD_DIM; index++) {
        key[index] = f32_to_bf16(random_signed() * amplitude);
        value[index] = f32_to_bf16(random_signed() * 2.0f);
    }
}

static int checked_row(const gqa_case *c, unsigned row) {
    return row % c->row_stride == 0 || row + 1 == c->sequence;
}

static void reference_attention(const gqa_case *c, const uint16_t *query,
                                const uint16_t *key, const uint16_t *value,
                                double *scores, float *output) {
    const unsigned QUERY_HEADS = c->query_heads, KV_HEADS = c->kv_heads;
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    for (unsigned row = 0; row < c->sequence; row++) {
        if (!checked_row(c, row)) continue;
        for (unsigned head = 0; head < QUERY_HEADS; head++) {
            unsigned kv_head = head / (QUERY_HEADS / KV_HEADS);
            double maximum = -INFINITY;
            for (unsigned key_row = 0; key_row <= row; key_row++) {
                double dot = 0.0;
                for (unsigned d = 0; d < HEAD_DIM; d++) {
                    size_t query_index =
                        ((size_t)row * QUERY_HEADS + head) * HEAD_DIM + d;
                    size_t key_index =
                        ((size_t)key_row * KV_HEADS + kv_head) * HEAD_DIM + d;
                    float scaled_query =
                        bf16_to_f32(query[query_index]) * scale;
                    dot = fma((double)scaled_query, bf16_to_f32(key[key_index]), dot);
                }
                scores[key_row] = dot;
                if (dot > maximum) maximum = dot;
            }
            double sum = 0.0;
            for (unsigned key_row = 0; key_row <= row; key_row++) {
                scores[key_row] = exp(scores[key_row] - maximum);
                sum += scores[key_row];
            }
            for (unsigned d = 0; d < HEAD_DIM; d++) {
                double result = 0.0;
                for (unsigned key_row = 0; key_row <= row; key_row++) {
                    size_t value_index =
                        ((size_t)key_row * KV_HEADS + kv_head) * HEAD_DIM + d;
                    result = fma(scores[key_row] / sum,
                                  bf16_to_f32(value[value_index]), result);
                }
                size_t output_index =
                    ((size_t)row * QUERY_HEADS + head) * HEAD_DIM + d;
                output[output_index] = (float)result;
            }
        }
    }
}

static void run_case(h3_gpu *gpu, const gqa_case *c) {
    const size_t query_count = (size_t)c->sequence * c->query_heads * HEAD_DIM;
    const size_t kv_count = (size_t)c->sequence * c->kv_heads * HEAD_DIM;
    const size_t row_elements = (size_t)c->query_heads * HEAD_DIM;
    uint16_t *query = malloc(query_count * sizeof(*query));
    uint16_t *key = malloc(kv_count * sizeof(*key));
    uint16_t *value = malloc(kv_count * sizeof(*value));
    uint16_t *got = malloc(query_count * sizeof(*got));
    float *reference = malloc(query_count * sizeof(*reference));
    double *scores = malloc(c->sequence * sizeof(*scores));
    require(query && key && value && got && reference && scores,
            "host allocation failed");
    if (c->tile_keys) {
        setenv("H3_GQA_FORCE_TILED", "1", 1);
        setenv("H3_GQA_TILE_KEYS", c->tile_keys, 1);
    } else {
        unsetenv("H3_GQA_FORCE_TILED");
        unsetenv("H3_GQA_TILE_KEYS");
    }
    unsetenv("H3_GQA_KERNEL");
    unsetenv("H3_MPS_GQA");
    if (c->path && !strcmp(c->path, "kernel")) setenv("H3_GQA_KERNEL", "1", 1);
    if (c->path && !strcmp(c->path, "mps")) setenv("H3_MPS_GQA", "1", 1);

    double squared_error = 0.0;
    double maximum_absolute = 0.0;
    size_t mismatches = 0;
    size_t elements = 0;
    for (unsigned trial = 0; trial < c->trials; trial++) {
        make_inputs(c, trial, query, key, value);
        reference_attention(c, query, key, value, scores, reference);
        h3_gpu_tensor *q = h3_gpu_tensor_from_bf16(gpu, query, query_count);
        h3_gpu_tensor *k = h3_gpu_tensor_from_bf16(gpu, key, kv_count);
        h3_gpu_tensor *v = h3_gpu_tensor_from_bf16(gpu, value, kv_count);
        h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, query_count);
        require(q && k && v && output, "Metal tensor allocation failed");
        require_gpu(gpu, h3_gpu_begin(gpu), "begin command stream");
        require_gpu(gpu, h3_gpu_gqa_causal_bf16(
            gpu, output, q, k, v, c->sequence, c->query_heads, c->kv_heads,
            HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)), "causal GQA");
        require_gpu(gpu, h3_gpu_submit(gpu), "submit command stream");
        require(h3_gpu_tensor_read_bf16(output, got, query_count),
                "cannot read GQA output");
        for (unsigned row = 0; row < c->sequence; row++) {
            if (!checked_row(c, row)) continue;
            for (size_t index = row * row_elements;
                 index < (row + 1) * row_elements; index++) {
                double delta = (double)bf16_to_f32(got[index]) - reference[index];
                double absolute = fabs(delta);
                if (absolute > maximum_absolute) maximum_absolute = absolute;
                squared_error += delta * delta;
                if (got[index] != f32_to_bf16(reference[index])) mismatches++;
            }
            elements += row_elements;
        }
        h3_gpu_tensor_free(output);
        h3_gpu_tensor_free(v);
        h3_gpu_tensor_free(k);
        h3_gpu_tensor_free(q);
    }

    double rmse = sqrt(squared_error / (double)elements);
    double mismatch_rate = (double)mismatches / (double)elements;
    printf("GQA %-24s: max abs %.7g, RMSE %.7g, BF16 mismatch %.5f%%\n",
           c->name, maximum_absolute, rmse, mismatch_rate * 100.0);
    require(maximum_absolute < 0.01, "GQA exceeds maximum error tolerance");
    require(rmse < 0.0015, "GQA exceeds RMSE tolerance");
    require(mismatch_rate < 0.001, "GQA exceeds BF16 mismatch tolerance");
    free(scores);
    free(reference);
    free(got);
    free(value);
    free(key);
    free(query);
}

int main(void) {
    char error[1024];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "FAIL tests/test_gqa.c: Metal setup: %s\n", error);
        return 1;
    }
    /* The 8200-token case exceeds the single-pass kernel's threadgroup score
     * row, as long Ref2VA multimodal prompts do, so it takes the tiled path. */
    /* Production routes prompts of 1024+ tokens to F32 MPSGraph SDPA and
     * splits them into causal query blocks above 12288 tokens. */
    static const gqa_case cases[] = {
        {"kernel single-pass S=32", 32, 8, 2, TRIALS, 1, NULL, NULL},
        {"kernel tiled(8) S=32", 32, 8, 2, TRIALS, 1, "8", "kernel"},
        {"kernel tiled S=8200", 8200, 4, 1, 2, 97, NULL, "kernel"},
        {"mps f32 S=32", 32, 8, 2, TRIALS, 1, NULL, "mps"},
        {"mps f32 S=8200", 8200, 4, 1, 2, 97, NULL, NULL},
        {"mps f32 split S=16500", 16500, 4, 1, 1, 131, NULL, NULL},
    };
    for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]); index++)
        run_case(gpu, &cases[index]);
    h3_gpu_free(gpu);
    puts("ok: causal GQA keeps F32 scaled queries; tiled path has no length cap");
    return 0;
}
