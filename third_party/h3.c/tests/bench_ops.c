#include "h3_gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

/* GPU throughput of the ops that dominate an H3 Ref2VA generation, at the
 * shapes they run at, for the Web UI self-test report. One JSON object per
 * line on stdout. Inputs are left uninitialized: timing only.
 * Usage: h3_bench [scale]  (scale 1 = default sizes, 0.5 = quicker). */

static h3_gpu *gpu;

static double now(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1e-6;
}

static void report(const char *op, const char *shape, double seconds,
                   double flop) {
    printf("{\"op\":\"%s\",\"shape\":\"%s\",\"seconds\":%.4f,"
           "\"tflops\":%.2f}\n", op, shape, seconds,
           seconds > 0 ? flop / seconds / 1e12 : 0.0);
    fflush(stdout);
}

static void fail(const char *op) {
    printf("{\"op\":\"%s\",\"error\":\"%s\"}\n", op, h3_gpu_error(gpu));
    fflush(stdout);
}

/* Best of `runs` timed runs after one warm-up. */
#define BENCH(op, shape, flop, runs, call) do {                              \
    double best = 1e30;                                                       \
    int ok = 1;                                                               \
    for (int run = 0; run <= (runs) && ok; run++) {                           \
        double start = now();                                                 \
        ok = h3_gpu_begin(gpu) && (call) && h3_gpu_submit(gpu);               \
        double elapsed = now() - start;                                       \
        if (run && elapsed < best) best = elapsed;                            \
    }                                                                         \
    if (ok) report(op, shape, best, flop); else fail(op);                    \
} while (0)

int main(int argc, char **argv) {
    double scale = argc > 1 ? atof(argv[1]) : 1.0;
    if (scale <= 0.0 || scale > 4.0) scale = 1.0;
    char error[512];
    gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        printf("{\"op\":\"setup\",\"error\":\"%s\"}\n", error);
        return 1;
    }
    char shape[128];

    /* DiT linears at a 16k-row slice of the joint sequence. */
    uint32_t rows = (uint32_t)(16384 * scale) & ~127u;
    h3_gpu_tensor *x = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * 28672);
    h3_gpu_tensor *w = h3_gpu_tensor_new_bf16(gpu, (size_t)28672 * 5376);
    h3_gpu_tensor *y = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * 28672);
    if (!x || !w || !y) { fail("gemm alloc"); return 1; }
    snprintf(shape, sizeof(shape), "%ux5376x5376", rows);
    BENCH("gemm_bf16", shape, 2.0 * rows * 5376.0 * 5376.0, 3,
          h3_gpu_linear_bf16(gpu, y, x, w, NULL, rows, 5376, 5376));
    snprintf(shape, sizeof(shape), "%ux5376x28672", rows);
    BENCH("gemm_bf16", shape, 2.0 * rows * 5376.0 * 28672.0, 3,
          h3_gpu_linear_bf16(gpu, y, x, w, NULL, rows, 5376, 28672));
    h3_gpu_tensor_free(x); h3_gpu_tensor_free(y); h3_gpu_tensor_free(w);

    /* DiT attention at an odd length (exercises padding + query split). */
    uint32_t sequence = ((uint32_t)(24576 * scale) & ~127u) + 1;
    size_t count = (size_t)sequence * 56 * 128;
    h3_gpu_tensor *q = h3_gpu_tensor_new_bf16(gpu, count);
    h3_gpu_tensor *k = h3_gpu_tensor_new_bf16(gpu, count);
    h3_gpu_tensor *v = h3_gpu_tensor_new_bf16(gpu, count);
    h3_gpu_tensor *o = h3_gpu_tensor_new_bf16(gpu, count);
    if (!q || !k || !v || !o) { fail("sdpa alloc"); return 1; }
    snprintf(shape, sizeof(shape), "S=%u heads=56 dim=128", sequence);
    BENCH("sdpa_bf16", shape, 4.0 * sequence * (double)sequence * 56 * 128, 2,
          h3_gpu_sdpa_bf16(gpu, o, q, k, v, sequence, 56, 128, 0.0883883f));

    /* Qwen causal GQA (64 query / 8 KV heads) at a Ref2VA prompt length. */
    uint32_t tokens = (uint32_t)(8192 * scale) + 7;
    snprintf(shape, sizeof(shape), "S=%u q=64 kv=8 dim=128", tokens);
    BENCH("gqa_causal_bf16", shape, 2.0 * tokens * (double)tokens * 64 * 128, 2,
          h3_gpu_gqa_causal_bf16(gpu, o, q, k, v, tokens, 64, 8, 128,
                                 0.0883883f));
    h3_gpu_tensor_free(q); h3_gpu_tensor_free(k);
    h3_gpu_tensor_free(v); h3_gpu_tensor_free(o);

    /* Video VAE encoder level-0 conv on one 256px tile. */
    uint32_t depth = (uint32_t)(33 * scale) | 1u, side = 256, channels = 128;
    size_t padded = (size_t)(depth + 2) * (side + 2) * (side + 2) * channels;
    size_t out = (size_t)depth * side * side * channels;
    h3_gpu_tensor *ci = h3_gpu_tensor_new_f32(gpu, padded);
    h3_gpu_tensor *cw = h3_gpu_tensor_new_f32(gpu, (size_t)channels * channels * 27);
    h3_gpu_tensor *co = h3_gpu_tensor_new_f32(gpu, out);
    if (!ci || !cw || !co) { fail("conv alloc"); return 1; }
    snprintf(shape, sizeof(shape), "%ux%ux%u %u->%u k3", depth, side, side,
             channels, channels);
    BENCH("conv3d_f32", shape,
          2.0 * depth * side * (double)side * channels * channels * 27, 2,
          h3_gpu_conv3d_f32(gpu, co, ci, cw, NULL, 1, depth + 2, side + 2,
                            side + 2, channels, channels, 3, 3, 3, 1, 1, 1));
    h3_gpu_tensor_free(ci); h3_gpu_tensor_free(cw); h3_gpu_tensor_free(co);
    h3_gpu_free(gpu);
    return 0;
}
