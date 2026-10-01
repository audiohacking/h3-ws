#include "h3_ref_cache.h"

#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

enum { H3_REF_CACHE_FORMAT = 1 };

static const char h3_ref_cache_magic[4] = {'H', '3', 'R', 'C'};

typedef struct {
    char magic[4];
    uint32_t format;
    uint32_t latent_t, latent_h, latent_w;
    uint32_t reserved;
    uint64_t elements;
    uint64_t key[2];
    uint64_t checksum;
} h3_ref_cache_header;

static uint64_t mix64(uint64_t value) {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ull;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

static uint64_t rotl64(uint64_t value, int bits) {
    return (value << bits) | (value >> (64 - bits));
}

/* Two independent 64-bit lanes over 8-byte words (plus a byte tail). Not
 * cryptographic: the cache trusts its own directory, it only needs inputs
 * that differ to land on different keys. */
static void hash_bytes(const void *data, size_t bytes, uint64_t lanes[2]) {
    const unsigned char *cursor = data;
    uint64_t a = lanes[0], b = lanes[1];
    size_t words = bytes / 8;
    for (size_t index = 0; index < words; index++) {
        uint64_t word;
        memcpy(&word, cursor + index * 8, 8);
        a = rotl64(a ^ (word * 0x9e3779b97f4a7c15ull), 31) * 0xff51afd7ed558ccdull;
        b = rotl64(b + word, 27) * 0xc4ceb9fe1a85ec53ull + 0x52dce729ull;
    }
    for (size_t index = words * 8; index < bytes; index++) {
        a = (a ^ cursor[index]) * 0x100000001b3ull;
        b = (b + cursor[index]) * 0x9e3779b97f4a7c15ull;
    }
    lanes[0] = a;
    lanes[1] = b;
}

h3_ref_cache_key h3_ref_cache_key_for(const float *pixels, int frames,
                                      int height, int width,
                                      const char *encoder_identity) {
    uint64_t lanes[2] = {0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull};
    int32_t shape[3] = {frames, height, width};
    hash_bytes(shape, sizeof(shape), lanes);
    if (encoder_identity)
        hash_bytes(encoder_identity, strlen(encoder_identity), lanes);
    size_t count = (size_t)3 * (size_t)frames * (size_t)height * (size_t)width;
    if (pixels) hash_bytes(pixels, count * sizeof(*pixels), lanes);
    h3_ref_cache_key key = {{mix64(lanes[0] ^ count), mix64(lanes[1] + count)}};
    return key;
}

const char *h3_ref_cache_dir(void) {
    const char *directory = getenv("H3_REF_CACHE_DIR");
    return directory && *directory ? directory : NULL;
}

static uint64_t payload_checksum(const float *rows, size_t elements) {
    uint64_t lanes[2] = {0x3c6ef372fe94f82bull, 0xa54ff53a5f1d36f1ull};
    hash_bytes(rows, elements * sizeof(*rows), lanes);
    return mix64(lanes[0]) ^ lanes[1];
}

static int entry_path(char *path, size_t size, const char *directory,
                      const h3_ref_cache_key *key) {
    int written = snprintf(path, size, "%s/%016llx%016llx.h3rc", directory,
                           (unsigned long long)key->hash[0],
                           (unsigned long long)key->hash[1]);
    return written > 0 && (size_t)written < size;
}

int h3_ref_cache_load(const char *directory, const h3_ref_cache_key *key,
                      int latent_t, int latent_h, int latent_w,
                      float *rows, size_t elements) {
    char path[1024];
    if (!directory || !key || !rows || !elements ||
        !entry_path(path, sizeof(path), directory, key)) return 0;
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    h3_ref_cache_header header;
    float *payload = NULL;
    int ok = fread(&header, sizeof(header), 1, file) == 1 &&
        !memcmp(header.magic, h3_ref_cache_magic, 4) &&
        header.format == H3_REF_CACHE_FORMAT &&
        header.latent_t == (uint32_t)latent_t &&
        header.latent_h == (uint32_t)latent_h &&
        header.latent_w == (uint32_t)latent_w &&
        header.elements == elements &&
        header.key[0] == key->hash[0] && header.key[1] == key->hash[1];
    if (ok) {
        payload = malloc(elements * sizeof(*payload));
        ok = payload && fread(payload, sizeof(*payload), elements, file) ==
                            elements &&
             fgetc(file) == EOF &&
             payload_checksum(payload, elements) == header.checksum;
    }
    fclose(file);
    if (ok) {
        memcpy(rows, payload, elements * sizeof(*rows));
        utimes(path, NULL);  /* most recently used, for pruning */
    }
    free(payload);
    return ok;
}

static int make_directories(const char *directory) {
    char path[1024];
    if (snprintf(path, sizeof(path), "%s", directory) >= (int)sizeof(path))
        return 0;
    for (char *cursor = path + 1; *cursor; cursor++) {
        if (*cursor != '/') continue;
        *cursor = '\0';
        if (mkdir(path, 0755) != 0 && errno != EEXIST) return 0;
        *cursor = '/';
    }
    return mkdir(path, 0755) == 0 || errno == EEXIST;
}

static uint64_t cache_max_bytes(void) {
    const char *value = getenv("H3_REF_CACHE_MAX_GB");
    double gigabytes = value && *value ? atof(value) : 20.0;
    if (!(gigabytes > 0.0)) gigabytes = 20.0;
    return (uint64_t)(gigabytes * 1024.0 * 1024.0 * 1024.0);
}

int h3_ref_cache_store(const char *directory, const h3_ref_cache_key *key,
                       int latent_t, int latent_h, int latent_w,
                       const float *rows, size_t elements) {
    char path[1024], temporary[1100];
    if (!directory || !key || !rows || !elements ||
        !make_directories(directory) ||
        !entry_path(path, sizeof(path), directory, key)) return 0;
    snprintf(temporary, sizeof(temporary), "%s.tmp.%d", path, (int)getpid());
    h3_ref_cache_header header;
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, h3_ref_cache_magic, 4);
    header.format = H3_REF_CACHE_FORMAT;
    header.latent_t = (uint32_t)latent_t;
    header.latent_h = (uint32_t)latent_h;
    header.latent_w = (uint32_t)latent_w;
    header.elements = elements;
    header.key[0] = key->hash[0];
    header.key[1] = key->hash[1];
    header.checksum = payload_checksum(rows, elements);
    FILE *file = fopen(temporary, "wb");
    if (!file) return 0;
    int ok = fwrite(&header, sizeof(header), 1, file) == 1 &&
             fwrite(rows, sizeof(*rows), elements, file) == elements;
    ok = fclose(file) == 0 && ok;
    /* rename() is atomic: readers see the old entry, none, or all of it. */
    if (!ok || rename(temporary, path) != 0) {
        unlink(temporary);
        return 0;
    }
    h3_ref_cache_prune(directory, cache_max_bytes());
    return 1;
}

typedef struct {
    char name[96];
    uint64_t bytes;
    double modified;
} cache_entry;

static int compare_oldest(const void *left, const void *right) {
    double a = ((const cache_entry *)left)->modified;
    double b = ((const cache_entry *)right)->modified;
    return (a > b) - (a < b);
}

void h3_ref_cache_prune(const char *directory, uint64_t max_bytes) {
    DIR *handle = directory ? opendir(directory) : NULL;
    if (!handle) return;
    cache_entry *entries = NULL;
    size_t count = 0, capacity = 0;
    uint64_t total = 0;
    struct dirent *item;
    while ((item = readdir(handle))) {
        size_t length = strlen(item->d_name);
        if (length < 6 || length >= sizeof(entries->name) ||
            strcmp(item->d_name + length - 5, ".h3rc")) continue;
        char path[1200];
        snprintf(path, sizeof(path), "%s/%s", directory, item->d_name);
        struct stat status;
        if (stat(path, &status) != 0) continue;
        if (count == capacity) {
            size_t grown = capacity ? capacity * 2 : 64;
            cache_entry *bigger = realloc(entries, grown * sizeof(*entries));
            if (!bigger) break;
            entries = bigger;
            capacity = grown;
        }
        snprintf(entries[count].name, sizeof(entries[count].name), "%s",
                 item->d_name);
        entries[count].bytes = (uint64_t)status.st_size;
        entries[count].modified = (double)status.st_mtimespec.tv_sec +
                                  1e-9 * (double)status.st_mtimespec.tv_nsec;
        total += entries[count].bytes;
        count++;
    }
    closedir(handle);
    if (total > max_bytes && count) {
        qsort(entries, count, sizeof(*entries), compare_oldest);
        for (size_t index = 0; index < count && total > max_bytes; index++) {
            char path[1200];
            snprintf(path, sizeof(path), "%s/%s", directory, entries[index].name);
            if (unlink(path) == 0) total -= entries[index].bytes;
        }
    }
    free(entries);
}
