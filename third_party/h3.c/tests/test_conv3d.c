#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Conv3d (pre-padded NDHWC input, OIDHW weights) against a double CPU
 * oracle for every kernel/stride the video VAE encoder uses, through both
 * the native MPSGraph op and the per-temporal-tap Conv2d decomposition. */

typedef struct {
    const char *name;
    uint32_t depth, height, width, input_channels, output_channels;
    uint32_t kernel, stride_depth, stride_spatial;
    int bias;
} conv_case;

static uint32_t random_state = 0x2545f491u;

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_conv3d.c: %s\n", message);
    exit(1);
}

static void require_gpu(h3_gpu *gpu, int condition, const char *operation) {
    if (condition) return;
    fprintf(stderr, "FAIL tests/test_conv3d.c: %s: %s\n", operation,
            h3_gpu_error(gpu));
    exit(1);
}

static float random_signed(void) {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    return (float)((random_state >> 8) * (1.0 / 16777216.0) * 2.0 - 1.0);
}

static uint32_t out_extent(uint32_t extent, uint32_t kernel, uint32_t stride) {
    return (extent - kernel) / stride + 1;
}

static void reference(const conv_case *c, const float *input,
                      const float *weight, const float *bias, double *output) {
    uint32_t od = out_extent(c->depth, c->kernel, c->stride_depth);
    uint32_t oh = out_extent(c->height, c->kernel, c->stride_spatial);
    uint32_t ow = out_extent(c->width, c->kernel, c->stride_spatial);
    uint32_t k = c->kernel;
    for (uint32_t t = 0; t < od; t++)
        for (uint32_t y = 0; y < oh; y++)
            for (uint32_t x = 0; x < ow; x++)
                for (uint32_t o = 0; o < c->output_channels; o++) {
                    double sum = c->bias ? bias[o] : 0.0;
                    for (uint32_t kd = 0; kd < k; kd++)
                        for (uint32_t kh = 0; kh < k; kh++)
                            for (uint32_t kw = 0; kw < k; kw++)
                                for (uint32_t i = 0; i < c->input_channels; i++) {
                                    size_t in = ((((size_t)(t * c->stride_depth + kd) *
                                        c->height + y * c->stride_spatial + kh) *
                                        c->width + x * c->stride_spatial + kw) *
                                        c->input_channels) + i;
                                    size_t w = ((((size_t)o * c->input_channels + i) *
                                        k + kd) * k + kh) * k + kw;
                                    sum += (double)input[in] * weight[w];
                                }
                    output[(((size_t)t * oh + y) * ow + x) *
                           c->output_channels + o] = sum;
                }
}

static double run(h3_gpu *gpu, const conv_case *c, const float *input,
                  const float *weight, const float *bias,
                  const double *want, int native) {
    uint32_t od = out_extent(c->depth, c->kernel, c->stride_depth);
    uint32_t oh = out_extent(c->height, c->kernel, c->stride_spatial);
    uint32_t ow = out_extent(c->width, c->kernel, c->stride_spatial);
    size_t in_count = (size_t)c->depth * c->height * c->width *
                      c->input_channels;
    size_t w_count = (size_t)c->output_channels * c->input_channels *
                     c->kernel * c->kernel * c->kernel;
    size_t out_count = (size_t)od * oh * ow * c->output_channels;
    if (native) setenv("H3_CONV3D_NATIVE", "1", 1);
    else unsetenv("H3_CONV3D_NATIVE");
    h3_gpu_tensor *x = h3_gpu_tensor_from_f32(gpu, input, in_count);
    h3_gpu_tensor *w = h3_gpu_tensor_from_f32(gpu, weight, w_count);
    h3_gpu_tensor *b = c->bias ?
        h3_gpu_tensor_from_f32(gpu, bias, c->output_channels) : NULL;
    h3_gpu_tensor *y = h3_gpu_tensor_new_f32(gpu, out_count);
    float *got = malloc(out_count * sizeof(*got));
    if (!x || !w || (c->bias && !b) || !y || !got) die("allocation failed");
    require_gpu(gpu, h3_gpu_begin(gpu), "begin");
    require_gpu(gpu, h3_gpu_conv3d_f32(
        gpu, y, x, w, b, 1, c->depth, c->height, c->width,
        c->input_channels, c->output_channels, c->kernel, c->kernel,
        c->kernel, c->stride_depth, c->stride_spatial, c->stride_spatial),
        "conv3d");
    require_gpu(gpu, h3_gpu_submit(gpu), "submit");
    if (!h3_gpu_tensor_read_f32(y, got, out_count)) die("readback failed");
    double worst = 0.0;
    for (size_t index = 0; index < out_count; index++) {
        double error = fabs((double)got[index] - want[index]);
        if (error > worst) worst = error;
    }
    free(got);
    h3_gpu_tensor_free(y);
    h3_gpu_tensor_free(b);
    h3_gpu_tensor_free(w);
    h3_gpu_tensor_free(x);
    return worst;
}

int main(void) {
    static const conv_case cases[] = {
        {"3x3x3 stride 1 + bias", 9, 10, 12, 16, 24, 3, 1, 1, 1},
        {"3x3x3 stride 1", 7, 9, 9, 32, 8, 3, 1, 1, 0},
        {"3x3x3 stride (1,2,2)", 9, 11, 13, 16, 16, 3, 1, 2, 1},
        {"3x3x3 stride (2,2,2)", 11, 13, 11, 16, 16, 3, 2, 2, 1},
        {"1x1x1 shortcut", 5, 6, 7, 32, 48, 1, 1, 1, 1},
    };
    char error[512];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "FAIL tests/test_conv3d.c: Metal setup: %s\n", error);
        return 1;
    }
    for (size_t n = 0; n < sizeof(cases) / sizeof(cases[0]); n++) {
        const conv_case *c = &cases[n];
        size_t in_count = (size_t)c->depth * c->height * c->width *
                          c->input_channels;
        size_t w_count = (size_t)c->output_channels * c->input_channels *
                         c->kernel * c->kernel * c->kernel;
        size_t out_count = (size_t)out_extent(c->depth, c->kernel, c->stride_depth) *
            out_extent(c->height, c->kernel, c->stride_spatial) *
            out_extent(c->width, c->kernel, c->stride_spatial) *
            c->output_channels;
        float *input = malloc(in_count * sizeof(*input));
        float *weight = malloc(w_count * sizeof(*weight));
        float *bias = malloc(c->output_channels * sizeof(*bias));
        double *want = malloc(out_count * sizeof(*want));
        if (!input || !weight || !bias || !want) die("host allocation failed");
        for (size_t i = 0; i < in_count; i++) input[i] = random_signed();
        for (size_t i = 0; i < w_count; i++) weight[i] = random_signed() * 0.1f;
        for (uint32_t i = 0; i < c->output_channels; i++) bias[i] = random_signed();
        reference(c, input, weight, bias, want);
        double native = run(gpu, c, input, weight, bias, want, 1);
        double split = run(gpu, c, input, weight, bias, want, 0);
        printf("conv3d %-22s: native max abs %.3g, decomposed max abs %.3g\n",
               c->name, native, split);
        /* F32 accumulation over at most 27 * 32 products of magnitude <= 0.1. */
        if (native > 1e-4 || split > 1e-4) die("conv3d exceeds F32 tolerance");
        free(want);
        free(bias);
        free(weight);
        free(input);
    }
    h3_gpu_free(gpu);
    puts("ok: Conv2d-per-tap Conv3d matches the native op and a double oracle");
    return 0;
}
