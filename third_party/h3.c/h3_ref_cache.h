#ifndef H3_REF_CACHE_H
#define H3_REF_CACHE_H

#include <stddef.h>
#include <stdint.h>

/* Disk cache for reference-media video VAE latents (patchified condition
 * rows). Encoding a reference video dominates Ref2VA preprocessing, and its
 * result depends only on the decoded pixels and the encoder, not on prompt
 * or seed. Enabled by H3_REF_CACHE_DIR; H3_REF_CACHE_MAX_GB caps its size
 * (default 20, oldest entries pruned first). Every failure is a cache miss. */

typedef struct {
    uint64_t hash[2];
} h3_ref_cache_key;

/* Key over the exact F32 pixels the encoder would see, their shape, and the
 * encoder identity (weight directory plus numerics-changing switches). */
h3_ref_cache_key h3_ref_cache_key_for(const float *pixels, int frames,
                                      int height, int width,
                                      const char *encoder_identity);

/* Directory from H3_REF_CACHE_DIR, or NULL when the cache is disabled. */
const char *h3_ref_cache_dir(void);

/* Fill `rows` (exactly `elements` floats) from the cache. Returns 1 on a
 * verified hit, 0 otherwise (rows untouched on a miss). */
int h3_ref_cache_load(const char *directory, const h3_ref_cache_key *key,
                      int latent_t, int latent_h, int latent_w,
                      float *rows, size_t elements);

/* Store rows atomically and prune the directory to its size cap. Returns 1
 * when the entry was written. */
int h3_ref_cache_store(const char *directory, const h3_ref_cache_key *key,
                       int latent_t, int latent_h, int latent_w,
                       const float *rows, size_t elements);

/* Delete oldest entries until the directory holds at most `max_bytes`. */
void h3_ref_cache_prune(const char *directory, uint64_t max_bytes);

#endif
