#include "h3_ref_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

static int checks;

static void require(int condition, const char *message) {
    checks++;
    if (condition) return;
    fprintf(stderr, "FAIL tests/test_ref_cache.c: %s\n", message);
    exit(1);
}

static void entry_path(char *path, size_t size, const char *directory,
                       const h3_ref_cache_key *key) {
    snprintf(path, size, "%s/%016llx%016llx.h3rc", directory,
             (unsigned long long)key->hash[0], (unsigned long long)key->hash[1]);
}

static void set_age(const char *path, long seconds_ago) {
    struct timeval now;
    gettimeofday(&now, NULL);
    struct timeval times[2] = {{now.tv_sec - seconds_ago, 0},
                               {now.tv_sec - seconds_ago, 0}};
    utimes(path, times);
}

int main(void) {
    char directory[] = "/tmp/h3_ref_cache_test_XXXXXX";
    require(mkdtemp(directory) != NULL, "mkdtemp");
    char nested[256];
    snprintf(nested, sizeof(nested), "%s/a/b", directory);

    enum { FRAMES = 5, HEIGHT = 32, WIDTH = 48, LT = 2, LH = 2, LW = 3 };
    size_t pixel_count = (size_t)3 * FRAMES * HEIGHT * WIDTH;
    float *pixels = malloc(pixel_count * sizeof(*pixels));
    for (size_t i = 0; i < pixel_count; i++) pixels[i] = (float)(i % 251) / 251.0f;
    size_t elements = (size_t)LT * LH * LW / 4 * 96 + 7;
    float *rows = malloc(elements * sizeof(*rows));
    float *loaded = malloc(elements * sizeof(*loaded));
    for (size_t i = 0; i < elements; i++) rows[i] = (float)i * 0.25f - 3.0f;

    /* Keys: stable for identical input, different for any change. */
    h3_ref_cache_key key = h3_ref_cache_key_for(pixels, FRAMES, HEIGHT, WIDTH, "vae");
    h3_ref_cache_key same = h3_ref_cache_key_for(pixels, FRAMES, HEIGHT, WIDTH, "vae");
    require(!memcmp(&key, &same, sizeof(key)), "key is deterministic");
    pixels[pixel_count / 2] += 1e-6f;
    h3_ref_cache_key nudged = h3_ref_cache_key_for(pixels, FRAMES, HEIGHT, WIDTH, "vae");
    pixels[pixel_count / 2] -= 1e-6f;
    require(memcmp(&key, &nudged, sizeof(key)), "one pixel changes the key");
    h3_ref_cache_key other_vae = h3_ref_cache_key_for(pixels, FRAMES, HEIGHT, WIDTH, "vae2");
    require(memcmp(&key, &other_vae, sizeof(key)), "encoder identity changes the key");
    h3_ref_cache_key reshaped = h3_ref_cache_key_for(pixels, FRAMES, WIDTH, HEIGHT, "vae");
    require(memcmp(&key, &reshaped, sizeof(key)), "shape changes the key");

    /* Round trip, creating missing parent directories. */
    require(!h3_ref_cache_load(nested, &key, LT, LH, LW, loaded, elements),
            "cold cache misses");
    require(h3_ref_cache_store(nested, &key, LT, LH, LW, rows, elements), "store");
    memset(loaded, 0, elements * sizeof(*loaded));
    require(h3_ref_cache_load(nested, &key, LT, LH, LW, loaded, elements), "hit");
    require(!memcmp(rows, loaded, elements * sizeof(*rows)), "rows are bit-exact");

    /* Anything inconsistent is a miss that leaves the destination untouched. */
    for (size_t i = 0; i < elements; i++) loaded[i] = 42.0f;
    require(!h3_ref_cache_load(nested, &key, LT, LH, LW + 1, loaded, elements),
            "latent shape mismatch misses");
    require(!h3_ref_cache_load(nested, &key, LT, LH, LW, loaded, elements - 1),
            "element count mismatch misses");
    require(loaded[0] == 42.0f && loaded[elements - 1] == 42.0f,
            "a miss leaves rows untouched");

    char path[512];
    entry_path(path, sizeof(path), nested, &key);
    FILE *file = fopen(path, "r+b");
    fseek(file, -12, SEEK_END);
    int byte = fgetc(file);
    fseek(file, -12, SEEK_END);
    fputc(byte ^ 0x10, file);
    fclose(file);
    require(!h3_ref_cache_load(nested, &key, LT, LH, LW, loaded, elements),
            "a flipped payload bit fails the checksum");
    require(loaded[3] == 42.0f, "corrupt entry leaves rows untouched");

    require(h3_ref_cache_store(nested, &key, LT, LH, LW, rows, elements), "re-store");
    truncate(path, 200);
    require(!h3_ref_cache_load(nested, &key, LT, LH, LW, loaded, elements),
            "a truncated entry misses");

    require(h3_ref_cache_store(nested, &key, LT, LH, LW, rows, elements), "re-store 2");
    char moved[512];
    entry_path(moved, sizeof(moved), nested, &other_vae);
    rename(path, moved);
    require(!h3_ref_cache_load(nested, &other_vae, LT, LH, LW, loaded, elements),
            "an entry renamed to another key misses");
    unlink(moved);

    /* Pruning drops the least recently used entries first. */
    h3_ref_cache_key keys[3] = {key, nudged, other_vae};
    char paths[3][512];
    for (int i = 0; i < 3; i++) {
        require(h3_ref_cache_store(nested, &keys[i], LT, LH, LW, rows, elements),
                "store for pruning");
        entry_path(paths[i], sizeof(paths[i]), nested, &keys[i]);
    }
    set_age(paths[0], 300);
    set_age(paths[1], 200);
    set_age(paths[2], 100);
    /* A hit refreshes the oldest entry, so the middle one goes first. */
    require(h3_ref_cache_load(nested, &keys[0], LT, LH, LW, loaded, elements),
            "hit refreshes entry age");
    struct stat status;
    stat(paths[0], &status);
    h3_ref_cache_prune(nested, (uint64_t)status.st_size * 2);
    require(access(paths[1], F_OK) != 0, "least recently used entry pruned");
    require(access(paths[0], F_OK) == 0 && access(paths[2], F_OK) == 0,
            "recent entries kept");

    unsetenv("H3_REF_CACHE_DIR");
    require(h3_ref_cache_dir() == NULL, "cache disabled without H3_REF_CACHE_DIR");
    setenv("H3_REF_CACHE_DIR", nested, 1);
    require(h3_ref_cache_dir() && !strcmp(h3_ref_cache_dir(), nested),
            "H3_REF_CACHE_DIR enables the cache");

    for (int i = 0; i < 3; i++) unlink(paths[i]);
    char cleanup[600];
    snprintf(cleanup, sizeof(cleanup), "rm -rf '%s'", directory);
    system(cleanup);
    free(pixels); free(rows); free(loaded);
    printf("ok: reference latent cache, %d checks\n", checks);
    return 0;
}
