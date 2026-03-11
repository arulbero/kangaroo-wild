/*
 * kangaroo_wild.c
 * Pollard's kangaroo wild search using scored DP database
 *
 * Author: arulbero
 * License: GPL-3.0 (see LICENSE file)
 *
 * Uses modular inversion code by Jean Luc Pons (GPL-3.0)
 * Scored DP technique based on: D.J. Bernstein and T. Lange,
 * "Computing small discrete logarithms faster", INDOCRYPT 2012
 */


#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/stat.h>
#include <math.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <signal.h>
#include <stddef.h>

#include "G.h"
#include "fast_inv.h"

// ==========================================
// USER-CONFIGURABLE PARAMETERS
// File paths must match the tame DB output files.
// Performance tuning defaults
// ==========================================

// Performance tuning defaults
#define DEFAULT_NUM_WORKERS 16
#define DEFAULT_BATCH_K     20
#define VITA_WILD_MAX   (1ULL << 23)  // Max steps before wild respawn

// File paths (must match the tame DB output files)

static const char* DB_FILENAME = "75_scored_16_tame_db_v2.bin";
static const char* FINGERPRINT_FILENAME = "75_scored_16_fingerprints_v2.bin";
static const char* BUCKET_OFFSETS_FILENAME = "75_scored_16_bucket_offsets_v2.bin";
static const char* TRAINING_PARAMS_FILENAME = "75_training_16_params_v2.bin";


static const char* PUBKEY_LIST_FILENAME = "public_keys.txt";

// ==========================================
// ALL REMAINING PARAMETERS ARE AUTO-DETECTED
// FROM training_params.bin AND THE DB FILES.
// DO NOT EDIT BELOW THIS LINE.
// ==========================================

#define MODE_WILD 1

// Compile-time maximums (for stack-allocated arrays — hardware-dependent only)
#define MAX_BATCH_K          DEFAULT_BATCH_K
#define MAX_NUM_WORKERS      DEFAULT_NUM_WORKERS
#define MAX_DIST_BYTES       16    // supports up to 128-bit ranges

// Runtime worker/batch params
static int NUM_WORKERS;
static int BATCH_K;

// ==========================================
// RUNTIME PARAMETERS (loaded from DB files)
// ==========================================

// From training_params.bin
static uint32_t rt_global_bits       = 0;
static uint32_t rt_local_bits        = 0;
static uint32_t rt_range_bits_low    = 0;
static uint32_t rt_range_bits_high   = 0;
static uint32_t rt_dist_bytes        = 0;
static uint32_t rt_jump_table_seed   = 42;
static uint64_t rt_scored_target_dp  = 0;
static uint32_t rt_jump_table_bits   = 0;
static uint32_t rt_jump_table_size   = 0;
static uint32_t rt_history_size      = 0;
static uint32_t rt_escape_table_size = 0;
static uint32_t rt_escape_mult       = 0;
static uint32_t rt_min_dp_steps      = 0;
static uint32_t rt_local_buf1_size   = 0;
static uint32_t rt_local_buf1_mask   = 0;
static uint32_t rt_trunc_bits        = 0;  // 0 = no truncation (legacy)

// From bucket_offsets file size
static uint32_t rt_hash_index_bits   = 0;
static uint64_t rt_hash_index_size   = 0;
static uint64_t rt_hash_index_mask   = 0;

// Runtime life limit — reduced in extend mode for faster partition rotation
static uint64_t rt_life_limit = VITA_WILD_MAX;
static double   g_q_hat    = 0;
static double   g_R_factor = 0;
static uint64_t g_N_sel    = 0;

// ==========================================
// STRUCT DEFINITIONS
// ==========================================

// CompactDPEntry uses MAX size on stack; only rt_dist_bytes are meaningful.
#pragma pack(push, 1)
typedef struct {
    uint8_t dist[MAX_DIST_BYTES];
} CompactDPEntry;
#pragma pack(pop)

typedef struct { uint64_t x[4]; uint64_t y[4]; } Point;
typedef struct { int64_t dist; Point pt; } JumpEntry;
typedef struct { int wid; int mode; int partition_id; } WorkerData;

// ==========================================
// GLOBAL STATE
// ==========================================

static JumpEntry* jump_table = NULL;
static JumpEntry* escape_table = NULL;
static Point target_point, wild_base;
static volatile int shutdown_flag = 0;
static volatile int sigint_received = 0;
static volatile int found_flag = 0;
static volatile int found_partition = 0;
static uint64_t found_key[4];

static CompactDPEntry* compact_db_array = NULL;
static uint64_t tame_db_count = 0;
static int tame_db_fd = -1;
static size_t tame_db_file_size = 0;

static pthread_mutex_t db_mutex = PTHREAD_MUTEX_INITIALIZER;
static volatile uint64_t worker_steps[256 * 8];
static volatile int worker_done[256] = {0};

static volatile uint64_t wild_db_matches = 0;
static volatile uint64_t total_lookups = 0;
static volatile uint64_t disk_lookups = 0;
static volatile uint64_t stat_fp_dupes = 0;

static volatile uint64_t stat_history_cycles = 0;
static volatile uint64_t stat_history_same_cycle = 0;
static volatile uint64_t stat_escape_respawn = 0;
static volatile uint64_t stat_buf1_respawn = 0;
static volatile uint64_t stat_life_respawn = 0;
static volatile uint64_t stat_visited_dp_cycle = 0;

static __int128 w_quarto, w_half, midpoint;

// Extended range mode: per-partition targets
#define MAX_PARTITIONS 4096   // supports up to 12 extra bits over DB range
static int num_partitions = 1;  // 1 = normal mode (no partitioning)
static Point partition_target[MAX_PARTITIONS];
static Point partition_wild_base[MAX_PARTITIONS];
static unsigned __int128 partition_offset[MAX_PARTITIONS];  // k * db_range_size

// ==========================================
// PARALLEL MODE STATE
// ==========================================
static volatile int parallel_mode = 0;
static volatile int partition_solved[MAX_PARTITIONS];
static uint64_t partition_found_key[MAX_PARTITIONS][4];
static volatile int num_unsolved = 0;
static char* partition_pubkey_hex[MAX_PARTITIONS];  // for reporting

static const uint64_t Gx[4] = {0x59f2815b16f81798, 0x029bfcdb2dce28d9, 0x55a06295ce870b07, 0x79be667ef9dcbbac};
static const uint64_t Gy[4] = {0x9c47d08ffb10d4b8, 0xfd17b448a6855419, 0x5da4fbfc0e1108a8, 0x483ada7726a3c465};

// Fingerprint / bucket offsets
static uint32_t* fingerprints = NULL;
static void* bucket_offsets = NULL;
static int bucket_offsets_is32 = 0;
static size_t bucket_offsets_map_size = 0;
static uint64_t fingerprint_count = 0;

// ==========================================
// INSTRUMENTATION
// ==========================================

typedef struct {
    uint64_t total_steps;
    uint64_t count_life, count_buf1, count_escape;
    uint64_t life_at_respawn_sum, life_at_respawn_count;
    uint64_t escape_jumps;
    uint64_t dp_hits_total, dp_filtered, dp_deduped, dp_saved;
    uint64_t wild_dp_checked, wild_dp_matched;
    uint64_t padding[4];
} WorkerStats;

static WorkerStats wstats[256];

// ==========================================
// TRAINING PARAMS (V1 + V2)
// V1: original fields (always present)
// V2: extended fields with all internal params
// ==========================================

#define TRAINING_PARAMS_MAGIC 0x5452414E5041524DULL  // "TRANPARM"

#pragma pack(push, 1)
typedef struct {
    // V1 fields (always present after magic)
    uint32_t global_bits;
    uint32_t range_bits_low;
    uint32_t range_bits_high;
    uint32_t jump_table_seed;
    uint64_t scored_target_dp;
    double   q_hat;
    double   R_factor;
    uint64_t N_sel;
    uint64_t reserved1;
    // V2 fields (may not be present in old files)
    uint32_t local_bits;
    uint32_t jump_table_bits;
    uint32_t history_size;
    uint32_t escape_table_size;
    uint32_t escape_mult;
    uint32_t min_dp_steps;
    uint32_t hash_index_bits;
    uint32_t trunc_bits;         // TRUNC_BITS (0 = no truncation, legacy)
} TrainingParamsV2;
#pragma pack(pop)

// Size of V1 payload (after magic): 4+4+4+4+8+8+8+8+8 = 56 bytes
#define TP_V1_PAYLOAD_SIZE 56
// V2 extension: 7 * 4 + 4 pad = 32 bytes
#define TP_V2_EXT_SIZE     32

static int load_training_params(void) {
    FILE* f = fopen(TRAINING_PARAMS_FILENAME, "rb");
    if (!f) {
        printf("[ERROR] Cannot open training_params file: %s\n", TRAINING_PARAMS_FILENAME);
        return 0;
    }

    // Read and verify magic
    uint64_t magic;
    if (fread(&magic, sizeof(uint64_t), 1, f) != 1 || magic != TRAINING_PARAMS_MAGIC) {
        printf("[ERROR] Invalid magic in training_params file\n");
        fclose(f);
        return -1;
    }

    // Read payload
    TrainingParamsV2 tp;
    memset(&tp, 0, sizeof(tp));
    size_t read_bytes = fread(&tp, 1, sizeof(TrainingParamsV2), f);
    fclose(f);

    if (read_bytes < TP_V1_PAYLOAD_SIZE) {
        printf("[ERROR] Training params file too small (%zu bytes payload, need %d)\n",
               read_bytes, TP_V1_PAYLOAD_SIZE);
        return -1;
    }

    // --- Load V1 fields (always present) ---
    rt_global_bits      = tp.global_bits;
    rt_range_bits_low   = tp.range_bits_low;
    rt_range_bits_high  = tp.range_bits_high;
    rt_jump_table_seed  = tp.jump_table_seed;
    rt_scored_target_dp = tp.scored_target_dp;
    g_q_hat             = tp.q_hat;
    g_R_factor          = tp.R_factor;
    g_N_sel             = tp.N_sel;

    // Derived: dist_bytes
    rt_dist_bytes = (rt_range_bits_high + 7) / 8;
    if (rt_dist_bytes > MAX_DIST_BYTES) {
        printf("[ERROR] RANGE_BITS_HIGH=%u needs %u dist bytes (max %d)\n",
               rt_range_bits_high, rt_dist_bytes, MAX_DIST_BYTES);
        return -1;
    }

    // --- Load V2 fields if present, otherwise use safe defaults ---
    if (read_bytes >= TP_V1_PAYLOAD_SIZE + TP_V2_EXT_SIZE && tp.local_bits > 0) {
        rt_local_bits        = tp.local_bits;
        rt_jump_table_bits   = tp.jump_table_bits;
        rt_jump_table_size   = 1U << tp.jump_table_bits;
        rt_history_size      = tp.history_size;
        rt_escape_table_size = tp.escape_table_size;
        rt_escape_mult       = tp.escape_mult;
        rt_min_dp_steps      = tp.min_dp_steps;
        rt_trunc_bits        = tp.trunc_bits;  // 0 in old V2 files (was _pad_v2)
        if (tp.hash_index_bits > 0) {
            rt_hash_index_bits = tp.hash_index_bits;
            rt_hash_index_size = 1ULL << tp.hash_index_bits;
            rt_hash_index_mask = rt_hash_index_size - 1;
        }
        printf("[PARAMS] V2 loaded: local=%u jump=%u hist=%u esc=%u mult=%u min_dp=%u trunc=%u\n",
               rt_local_bits, rt_jump_table_bits, rt_history_size,
               rt_escape_table_size, rt_escape_mult, rt_min_dp_steps, rt_trunc_bits);
    } else {
        // V1 file — use hardcoded defaults matching the original design
        rt_local_bits        = 8;
        rt_jump_table_bits   = 9;
        rt_jump_table_size   = 512;
        rt_history_size      = 4;
        rt_escape_table_size = 128;
        rt_escape_mult       = 2000;
        rt_min_dp_steps      = 1;
        rt_trunc_bits        = 0;  // legacy: no truncation
        printf("[PARAMS] V1 format — using defaults: local=8 jump=9 hist=4 esc=128 mult=2000 trunc=0\n");
    }

    // Recalculate dist_bytes accounting for truncation
    if (rt_trunc_bits > 0)
        rt_dist_bytes = (rt_range_bits_high - rt_trunc_bits + 7) / 8;

    // Allocate jump/escape tables now that we know sizes
    jump_table = calloc(rt_jump_table_size, sizeof(JumpEntry));
    escape_table = calloc(rt_escape_table_size, sizeof(JumpEntry));
    if (!jump_table || !escape_table) {
        printf("[ERROR] Cannot allocate jump/escape tables\n");
        return -1;
    }

    // Derived params
    rt_local_buf1_size = ((1U << (rt_global_bits - rt_local_bits)) * 4);
    rt_local_buf1_mask = rt_local_buf1_size - 1;

    printf("[PARAMS] GLOBAL_BITS=%u RANGE=%u-%u DIST_BYTES=%u TRUNC_BITS=%u\n",
           rt_global_bits, rt_range_bits_low, rt_range_bits_high, rt_dist_bytes, rt_trunc_bits);
    printf("[PARAMS] scored_target_dp=%lu seed=%u local_buf1=%u\n",
           (unsigned long)rt_scored_target_dp, rt_jump_table_seed, rt_local_buf1_size);
    if (g_q_hat > 0)
        printf("[PARAMS] q_hat=%.6f R_factor=%.1f N_sel=%lu\n",
               g_q_hat, g_R_factor, (unsigned long)g_N_sel);
    return 1;
}

// ==========================================
// INLINE HELPERS
// ==========================================

static inline __int128 load_compact_dist(const CompactDPEntry* e) {
    unsigned __int128 val = 0;
    for (uint32_t i = 0; i < rt_dist_bytes; i++)
        val |= (unsigned __int128)e->dist[i] << (i * 8);
    if (e->dist[rt_dist_bytes - 1] & 0x80)
        for (uint32_t i = rt_dist_bytes; i < 16; i++)
            val |= (unsigned __int128)0xFF << (i * 8);
    // Shift back: low rt_trunc_bits were truncated during DB generation
    return (__int128)val << rt_trunc_bits;
}

static inline uint32_t hash_fingerprint32(uint64_t x_hi) {
    uint64_t h = x_hi * 0xc6a4a7935bd1e995ULL;
    h ^= h >> 32;
    h *= 0x9e3779b97f4a7c15ULL;
    h ^= h >> 29;
    return (uint32_t)(h & 0xFFFFFFFF);
}

static inline uint64_t hash_x_hi(uint64_t x_hi) {
    uint64_t h = x_hi * 0x9e3779b97f4a7c15ULL;
    h ^= h >> 33;
    h *= 0xc6a4a7935bd1e995ULL;
    h ^= h >> 29;
    return h;
}

static inline uint64_t hash_for_bucket(uint64_t x_hi) {
    return hash_x_hi(x_hi) & rt_hash_index_mask;
}

static inline int cmp256(const uint64_t* a, const uint64_t* b) {
    for (int i = 3; i >= 0; i--) { if (a[i] > b[i]) return 1; if (a[i] < b[i]) return -1; }
    return 0;
}

static const uint64_t p_half[4] = {0xffffffff7ffffe17, 0xffffffffffffffff, 0xffffffffffffffff, 0x7fffffffffffffff};
static inline int y_is_greater_than_half(const uint64_t* y) {
    if (y[3] > 0x7fffffffffffffffULL) return 1;
    if (y[3] < 0x7fffffffffffffffULL) return 0;
    return cmp256(y, p_half) > 0;
}
static inline void point_copy(Point* dst, const Point* src) { memcpy(dst, src, sizeof(Point)); }
static inline int point_is_zero(const Point* p) { return !p->x[0] && !p->x[1] && !p->x[2] && !p->x[3]; }

static void scalar_mult(Point* result, const uint64_t* k, const Point* P) {
    uint64_t jx[4], jy[4], jz[4], kp[4]; memcpy(kp, k, 32);
    mul(jx, jy, jz, kp);
    uint64_t invz[4]; fast_modinv(invz, jz);
    jac_to_aff(result->x, result->y, jx, jy, invz);
}

static void point_add(Point* result, const Point* P, const Point* Q) {
    if (point_is_zero(P)) { point_copy(result, Q); return; }
    if (point_is_zero(Q)) { point_copy(result, P); return; }
    uint64_t jax[4], jay[4], jaz[4] = {1,0,0,0};
    memcpy(jax, P->x, 32); memcpy(jay, P->y, 32);
    aecc_add_ja(jax, jay, jaz, Q->x, Q->y);
    uint64_t invz[4]; fast_modinv(invz, jaz);
    jac_to_aff(result->x, result->y, jax, jay, invz);
}

static void point_neg(Point* result, const Point* P) {
    memcpy(result->x, P->x, 32);
    aecc_sub(result->y, pp, P->y);
}

static void batch_inv(uint64_t invs[][4], uint64_t vals[][4], int n) {
    if (n == 0) return;
    if (n == 1) { fast_modinv(invs[0], vals[0]); return; }
    uint64_t products[MAX_BATCH_K][4]; memcpy(products[0], vals[0], 32);
    for (int i = 1; i < n; i++) aecc_mul(products[i], products[i-1], vals[i]);
    uint64_t inv_all[4]; fast_modinv(inv_all, products[n-1]);
    for (int i = n-1; i > 0; i--) {
        aecc_mul(invs[i], inv_all, products[i-1]);
        aecc_mul(inv_all, inv_all, vals[i]);
    }
    memcpy(invs[0], inv_all, 32);
}

static void batch_point_add(Point* points, __int128* dists, const int* jump_idx, int n, JumpEntry* jt) {
    uint64_t dx[MAX_BATCH_K][4], inv_dx[MAX_BATCH_K][4];
    for (int i = 0; i < n; i++) aecc_sub(dx[i], jt[jump_idx[i]].pt.x, points[i].x);
    batch_inv(inv_dx, dx, n);
    for (int i = 0; i < n; i++) {
        uint64_t dy[4], lambda[4], lambda2[4], new_x[4], new_y[4], tmp[4];
        aecc_sub(dy, jt[jump_idx[i]].pt.y, points[i].y);
        aecc_mul(lambda, dy, inv_dx[i]);
        aecc_sqr(lambda2, lambda);
        aecc_sub(tmp, lambda2, points[i].x);
        aecc_sub(new_x, tmp, jt[jump_idx[i]].pt.x);
        aecc_sub(tmp, points[i].x, new_x);
        aecc_mul(new_y, lambda, tmp);
        aecc_sub(new_y, new_y, points[i].y);
        memcpy(points[i].x, new_x, 32);
        memcpy(points[i].y, new_y, 32);
        dists[i] += (__int128)jt[jump_idx[i]].dist;
    }
}

static uint64_t rand64(unsigned int* seed) { return ((uint64_t)rand_r(seed) << 32) | rand_r(seed); }
static inline uint64_t xorshift64(uint64_t* s) { uint64_t x = *s; x ^= x << 13; x ^= x >> 7; x ^= x << 17; return *s = x; }

static __int128 rand_uniform_centered(__int128 half, uint64_t* rng) {
    unsigned __int128 r = ((unsigned __int128)xorshift64(rng) << 64) | xorshift64(rng);
    return (__int128)(r % ((unsigned __int128)half * 2 + 1)) - half;
}

static int computed_blinding_divisor(void) {
    double N  = (double)rt_scored_target_dp;
    double D  = (double)(1ULL << rt_global_bits);
    double Cw = (double)(NUM_WORKERS * BATCH_K);
    double vita = (double)VITA_WILD_MAX / 4.0;
    double bd = N * D / (Cw * vita);
    if (bd < 8.0) bd = 8.0;
    if (bd > 10000.0) bd = 10000.0;
    return (int)bd;
}

static __int128 rand_gaussian_centered(__int128 half, uint64_t* rng) {
    int bd = computed_blinding_divisor();
    __int128 d = rand_uniform_centered(w_half / bd, rng);
    return (d > half) ? half : (d < -half) ? -half : d;
}

// ==========================================
// JUMP TABLE INIT
// ==========================================

static void init_jump_table(void) {
    unsigned int seed = rt_jump_table_seed;
    long double high_val = powl(2.0L, rt_range_bits_high);
    long double low_val  = powl(2.0L, rt_range_bits_low);
    double W = (double)(high_val - low_val) / 2.0;

    double gap = W / (double)rt_scored_target_dp;
    uint64_t opt = (uint64_t)(gap) / sqrt((double)(1ULL << rt_global_bits));
    if (opt < 1) opt = 1;

    printf("[JUMP_TABLE] target_dp=%lu gap=%.2e opt=%lu (2^%.1f)\n",
           (unsigned long)rt_scored_target_dp, gap, (unsigned long)opt,
           opt > 0 ? log2((double)opt) : 0);

    Point G; memcpy(G.x, Gx, 32); memcpy(G.y, Gy, 32);
    for (uint32_t i = 0; i < rt_jump_table_size; i++) {
        uint64_t d = (opt / 2) + (rand64(&seed) % opt);
        if (d == 0) d = 1;
        jump_table[i].dist = d;
        uint64_t k[4] = {d, 0, 0, 0};
        scalar_mult(&jump_table[i].pt, k, &G);
    }
    for (uint32_t i = 0; i < rt_escape_table_size; i++) {
        uint64_t d = (opt * rt_escape_mult) + (rand64(&seed) % (opt * rt_escape_mult));
        escape_table[i].dist = d;
        uint64_t k[4] = {d, 0, 0, 0};
        scalar_mult(&escape_table[i].pt, k, &G);
    }
}

// ==========================================
// PUBLIC KEY PARSER
// ==========================================

static void parse_pubkey(const char* hex, Point* pt) {
    int prefix = (hex[1] == '3');
    for (int i = 0; i < 4; i++) { pt->x[3-i] = 0;
        for (int j = 0; j < 16; j++) { char c = hex[2+i*16+j];
            pt->x[3-i] = (pt->x[3-i] << 4) | ((c >= 'a') ? c-'a'+10 : (c >= 'A') ? c-'A'+10 : c-'0'); }}
    uint64_t x2[4], x3[4], y2[4]; aecc_sqr(x2, pt->x); aecc_mul(x3, x2, pt->x);
    unsigned __int128 sum = (unsigned __int128)x3[0] + 7; y2[0] = sum;
    sum = (unsigned __int128)x3[1] + (sum >> 64); y2[1] = sum;
    sum = (unsigned __int128)x3[2] + (sum >> 64); y2[2] = sum;
    sum = (unsigned __int128)x3[3] + (sum >> 64); y2[3] = sum;
    while (cmp256(y2, pp) >= 0) aecc_sub(y2, y2, pp);
    uint64_t exp[4] = {0xffffffffbfffff0c, 0xffffffffffffffff, 0xffffffffffffffff, 0x3fffffffffffffff};
    uint64_t res[4] = {1,0,0,0}, base[4]; memcpy(base, y2, 32);
    for (int i = 0; i < 256; i++) { if ((exp[i/64] >> (i%64)) & 1) aecc_mul(res, res, base); aecc_sqr(base, base); }
    memcpy(pt->y, res, 32);
    if ((prefix && !(pt->y[0] & 1)) || (!prefix && (pt->y[0] & 1))) aecc_sub(pt->y, pp, pt->y);
}

// ==========================================
// DB LOADING
// ==========================================

#define MAX_FP_MATCHES 8

static inline uint64_t bucket_off_get(uint64_t idx) {
    if (bucket_offsets_is32) return (uint64_t)((const uint32_t*)bucket_offsets)[idx];
    else return ((const uint64_t*)bucket_offsets)[idx];
}

static int tame_db_lookup_hash(uint64_t x_hi, __int128* dist_out, int* n_matches) {
    if (!tame_db_count || !bucket_offsets || !fingerprints) return 0;
    __sync_fetch_and_add(&total_lookups, 1);
    uint64_t bucket = hash_for_bucket(x_hi);
    uint32_t fp = hash_fingerprint32(x_hi);
    uint64_t start = bucket_off_get(bucket);
    uint64_t end   = bucket_off_get(bucket + 1);
    int count = 0;
    for (uint64_t i = start; i < end; i++) {
        if (fingerprints[i] == fp) {
            __sync_fetch_and_add(&disk_lookups, 1);
            if (dist_out && count < MAX_FP_MATCHES)
                dist_out[count] = load_compact_dist(&compact_db_array[i]);
            count++;
        }
    }
    if (n_matches) *n_matches = (count > MAX_FP_MATCHES) ? MAX_FP_MATCHES : count;
    return count > 0;
}

static void load_tame_db(const char* filename) {
    struct stat sb;
    if (stat(filename, &sb) == -1 || sb.st_size == 0) {
        printf("[DB] File not found or empty: %s\n", filename);
        tame_db_count = 0; return;
    }
    tame_db_fd = open(filename, O_RDONLY);
    tame_db_file_size = sb.st_size;

    // On-disk entries are rt_dist_bytes each; our struct is MAX_DIST_BYTES wide
    uint64_t entry_size_on_disk = rt_dist_bytes;
    tame_db_count = tame_db_file_size / entry_size_on_disk;

    printf("====================================================\n");
    printf("[DB] %.2f KB | %lu entries (disk: %lu B/entry, dist:%u, trunc:%u)\n",
           tame_db_file_size/1024.0, (unsigned long)tame_db_count,
           (unsigned long)entry_size_on_disk, rt_dist_bytes, rt_trunc_bits);

    if (rt_dist_bytes == MAX_DIST_BYTES) {
        compact_db_array = mmap(NULL, tame_db_file_size, PROT_READ, MAP_SHARED, tame_db_fd, 0);
        if (compact_db_array == MAP_FAILED) { printf("[ERROR] mmap DB failed\n"); exit(1); }
    } else {
        // Expand: read rt_dist_bytes per entry, zero-pad to MAX_DIST_BYTES
        uint8_t* raw = mmap(NULL, tame_db_file_size, PROT_READ, MAP_SHARED, tame_db_fd, 0);
        if (raw == MAP_FAILED) { printf("[ERROR] mmap DB failed\n"); exit(1); }
        compact_db_array = calloc(tame_db_count, sizeof(CompactDPEntry));
        if (!compact_db_array) { printf("[ERROR] alloc expanded DB failed\n"); exit(1); }
        for (uint64_t i = 0; i < tame_db_count; i++)
            memcpy(compact_db_array[i].dist, raw + i * entry_size_on_disk, rt_dist_bytes);
        munmap(raw, tame_db_file_size);
    }
    madvise(compact_db_array, tame_db_count * sizeof(CompactDPEntry), MADV_RANDOM);

    // Fingerprints
    struct stat fp_stat;
    if (stat(FINGERPRINT_FILENAME, &fp_stat) == -1) {
        printf("[ERROR] Fingerprint file not found: %s\n", FINGERPRINT_FILENAME); exit(1);
    }
    int fp_fd = open(FINGERPRINT_FILENAME, O_RDONLY);
    fingerprint_count = fp_stat.st_size / sizeof(uint32_t);
    fingerprints = mmap(NULL, fp_stat.st_size, PROT_READ, MAP_SHARED, fp_fd, 0);
    if (fingerprints == MAP_FAILED) { printf("[ERROR] mmap fingerprints failed\n"); exit(1); }
    mlock(fingerprints, fp_stat.st_size);
    printf("[FP] Fingerprints: %.2f KB (%lu entries)\n", fp_stat.st_size/1024.0, fingerprint_count);

    // Bucket offsets — auto-detect HASH_INDEX_BITS if not set by V2
    struct stat bo_stat;
    if (stat(BUCKET_OFFSETS_FILENAME, &bo_stat) == -1) {
        printf("[ERROR] Bucket offsets not found: %s\n", BUCKET_OFFSETS_FILENAME); exit(1);
    }
    int bo_fd = open(BUCKET_OFFSETS_FILENAME, O_RDONLY);
    bucket_offsets_map_size = (size_t)bo_stat.st_size;

    if (rt_hash_index_bits == 0) {
        // Auto-detect from file size
        uint64_t entries_32 = bucket_offsets_map_size / sizeof(uint32_t);
        uint64_t entries_64 = bucket_offsets_map_size / sizeof(uint64_t);
        uint64_t hi_size = 0;

        if (entries_32 > 1 && (entries_32 - 1) > 0 && ((entries_32 - 1) & (entries_32 - 2)) == 0) {
            hi_size = entries_32 - 1; bucket_offsets_is32 = 1;
        } else if (entries_64 > 1 && (entries_64 - 1) > 0 && ((entries_64 - 1) & (entries_64 - 2)) == 0) {
            hi_size = entries_64 - 1; bucket_offsets_is32 = 0;
        } else {
            printf("[ERROR] Cannot detect HASH_INDEX_BITS from bucket_offsets (%zu bytes)\n",
                   bucket_offsets_map_size);
            exit(1);
        }
        rt_hash_index_size = hi_size;
        rt_hash_index_mask = hi_size - 1;
        rt_hash_index_bits = 0;
        uint64_t tmp = hi_size;
        while (tmp > 1) { rt_hash_index_bits++; tmp >>= 1; }
    } else {
        // Validate V2 value against file
        size_t exp32 = (rt_hash_index_size + 1) * sizeof(uint32_t);
        size_t exp64 = (rt_hash_index_size + 1) * sizeof(uint64_t);
        if (bucket_offsets_map_size == exp32) bucket_offsets_is32 = 1;
        else if (bucket_offsets_map_size == exp64) bucket_offsets_is32 = 0;
        else {
            printf("[ERROR] Bucket offsets size (%zu) doesn't match HASH_INDEX_BITS=%u from training_params\n",
                   bucket_offsets_map_size, rt_hash_index_bits);
            exit(1);
        }
    }

    bucket_offsets = mmap(NULL, bucket_offsets_map_size, PROT_READ, MAP_SHARED, bo_fd, 0);
    if (bucket_offsets == MAP_FAILED) { printf("[ERROR] mmap bucket offsets failed\n"); exit(1); }
    mlock(bucket_offsets, bucket_offsets_map_size);
    printf("[BO] Bucket offsets: %.2f KB (HASH_INDEX_BITS=%u, %s)\n",
           bucket_offsets_map_size/1024.0, rt_hash_index_bits,
           bucket_offsets_is32 ? "u32" : "u64");
    printf("====================================================\n");
}

// ==========================================
// INSTRUMENTATION REPORT
// ==========================================

static void print_instrumentation_report(int num_workers) {
    WorkerStats agg = {0};
    for (int i = 0; i < num_workers; i++) {
        agg.total_steps += wstats[i].total_steps;
        agg.count_life += wstats[i].count_life;
        agg.count_buf1 += wstats[i].count_buf1;
        agg.count_escape += wstats[i].count_escape;
        agg.escape_jumps += wstats[i].escape_jumps;
        agg.life_at_respawn_sum += wstats[i].life_at_respawn_sum;
        agg.life_at_respawn_count += wstats[i].life_at_respawn_count;
        agg.dp_hits_total += wstats[i].dp_hits_total;
        agg.dp_filtered += wstats[i].dp_filtered;
        agg.dp_deduped += wstats[i].dp_deduped;
        agg.dp_saved += wstats[i].dp_saved;
        agg.wild_dp_checked += wstats[i].wild_dp_checked;
        agg.wild_dp_matched += wstats[i].wild_dp_matched;
    }
    printf("\n============================================================\n");
    printf("           STEP ACCOUNTING REPORT\n");
    printf("============================================================\n");
    printf(" Total steps:          %12.2fM\n", agg.total_steps/1e6);
    printf("------------------------------------------------------------\n");
    printf(" BREAKDOWN:\n");
    printf("   life_limit rsp:     %12lu\n", agg.count_life);
    printf("   buf1 (cycle) rsp:   %12lu\n", agg.count_buf1);
    printf("   escape rsp:         %12lu\n", agg.count_escape);
    printf("   escape jumps:       %12lu\n", agg.escape_jumps);
    if (agg.life_at_respawn_count > 0)
        printf("   avg life at rsp:    %12.0f\n",
               (double)agg.life_at_respawn_sum / agg.life_at_respawn_count);
    printf("------------------------------------------------------------\n");
    printf(" DP STATS:\n");
    printf("   DP global hits:     %12lu\n", agg.dp_hits_total);
    printf("   DP filtered (k):    %12lu\n", agg.dp_filtered);
    printf("   DP saved:           %12lu\n", agg.dp_saved);
    printf(" WILD STATS:\n");
    printf("   DP checked vs DB:   %12lu\n", agg.wild_dp_checked);
    printf("   DB matches:         %12lu\n", agg.wild_dp_matched);
    printf("============================================================\n");
}

// ==========================================
// RESPAWN POOL
// ==========================================
#define POOL_SIZE (MAX_BATCH_K * 2)

typedef struct { Point pt; __int128 dist; } PoolEntry;

static inline void pool_generate_one(PoolEntry* entry, uint64_t* rng, const Point* wb) {
    __int128 d = rand_gaussian_centered(w_half, rng);
    uint64_t k[4] = {0,0,0,0};
    __int128 ad = (d >= 0) ? d : -d;
    k[0] = (uint64_t)ad; k[1] = (uint64_t)(ad >> 64);
    Point G_pt; memcpy(G_pt.x, Gx, 32); memcpy(G_pt.y, Gy, 32);
    scalar_mult(&entry->pt, k, &G_pt);
    if (d < 0) point_neg(&entry->pt, &entry->pt);
    Point tmp; point_copy(&tmp, &entry->pt);
    point_add(&entry->pt, wb, &tmp);
    if (y_is_greater_than_half(entry->pt.y)) {
        aecc_sub(entry->pt.y, pp, entry->pt.y);
        d = -d;
    }
    entry->dist = d;
}

// ==========================================
// HELPER: pick an unsolved partition
// ==========================================

static inline int pick_unsolved_partition(uint64_t* rng, int npart) {
    if (npart == 1) return 0;
    // Try random first
    for (int attempt = 0; attempt < npart * 2; attempt++) {
        int p = (int)(xorshift64(rng) % (uint64_t)npart);
        if (!partition_solved[p]) return p;
    }
    // Fallback: linear scan
    for (int p = 0; p < npart; p++)
        if (!partition_solved[p]) return p;
    return -1;  // all solved
}

// ==========================================
// WORKER THREAD (WILD ONLY)
// ==========================================

#define VISITED_DP_CAP 1024

static void* worker_thread(void* arg) {
    WorkerData* data = (WorkerData*)arg;
    int wid = data->wid;
    int pid = data->partition_id;  // default partition (used when num_partitions==1)
    memset(&wstats[wid], 0, sizeof(WorkerStats));

    // Per-kangaroo partition assignment
    int kang_part[MAX_BATCH_K];
    for (int i = 0; i < BATCH_K; i++) kang_part[i] = pid;

    uint64_t life_limit = rt_life_limit;
    uint64_t rng = wid * 0x9e3779b97f4a7c15ULL + time(NULL) + wid * 12345;

    JumpEntry* local_jt = malloc(rt_jump_table_size * sizeof(JumpEntry));
    memcpy(local_jt, jump_table, sizeof(JumpEntry) * rt_jump_table_size);

    uint64_t g_mask     = (1ULL << rt_global_bits) - 1;
    uint64_t l_mask     = (1ULL << rt_local_bits) - 1;
    uint64_t table_mask = rt_jump_table_size - 1;
    uint64_t esc_mask   = rt_escape_table_size - 1;

    Point cx[MAX_BATCH_K]; __int128 cd[MAX_BATCH_K];
    int life[MAX_BATCH_K], jump_idx[MAX_BATCH_K], needs_respawn[MAX_BATCH_K];
    uint64_t steps_since_respawn[MAX_BATCH_K];
    for (int k = 0; k < BATCH_K; k++) steps_since_respawn[k] = 0;

    Point*    hist_pt_flat = calloc((size_t)BATCH_K * rt_history_size, sizeof(Point));
    __int128* hist_dist_flat = calloc((size_t)BATCH_K * rt_history_size, sizeof(__int128));
    int      hist_idx[MAX_BATCH_K];
    #define HIST_PT(i, h)   hist_pt_flat[(size_t)(i) * rt_history_size + (h)]
    #define HIST_DIST(i, h) hist_dist_flat[(size_t)(i) * rt_history_size + (h)]

    uint64_t last_canonical_x0[MAX_BATCH_K];
    uint64_t last_canonical_x1[MAX_BATCH_K];

    // local_buf1: dynamic allocation since size depends on runtime params
    uint64_t* local_buf1_flat = calloc((size_t)BATCH_K * rt_local_buf1_size, sizeof(uint64_t));
    int local_buf1_idx[MAX_BATCH_K];
    memset(local_buf1_idx, 0, sizeof(local_buf1_idx));

    uint16_t* generation = calloc(BATCH_K, sizeof(uint16_t));
    for (int i = 0; i < BATCH_K; i++) generation[i] = 1;

    int escape_idx_arr[MAX_BATCH_K];
    memset(escape_idx_arr, 0, sizeof(escape_idx_arr));
    int consecutive_escapes[MAX_BATCH_K];
    memset(consecutive_escapes, 0, sizeof(consecutive_escapes));

    uint64_t pend_x_hi[MAX_BATCH_K * 2]; __int128 pend_dist[MAX_BATCH_K * 2];
    int pend_part[MAX_BATCH_K * 2];
    int pend_cnt = 0;

    uint64_t visited_dp[MAX_BATCH_K][VISITED_DP_CAP];
    int visited_dp_cnt[MAX_BATCH_K];
    memset(visited_dp_cnt, 0, sizeof(visited_dp_cnt));

    Point G; memcpy(G.x, Gx, 32); memcpy(G.y, Gy, 32);

    // Respawn pool (only used in single-partition mode)
    PoolEntry* pool = malloc(POOL_SIZE * sizeof(PoolEntry));
    int pool_head = 0, pool_avail = POOL_SIZE;
    uint64_t pool_rng = rng ^ 0xA5A5A5A5A5A5A5A5ULL;
    if (num_partitions == 1) {
        for (int p = 0; p < POOL_SIZE; p++)
            pool_generate_one(&pool[p], &pool_rng, &partition_wild_base[pid]);
    } else {
        pool_avail = 0;  // disable pool for multi-partition modes
    }

    for (int i = 0; i < BATCH_K; i++) {
        needs_respawn[i] = 1; life[i] = 0; hist_idx[i] = 0;
        memset(&HIST_PT(i, 0), 0, sizeof(Point) * rt_history_size);
        memset(&HIST_DIST(i, 0), 0, sizeof(__int128) * rt_history_size);
        last_canonical_x0[i] = 0; last_canonical_x1[i] = 0;
    }

    uint64_t steps = 0;

    // Access local_buf1 for slot i, index j
    #define LB1(i, j) local_buf1_flat[(size_t)(i) * rt_local_buf1_size + (j)]

    while (steps < (1ULL << 60) && !shutdown_flag) {

        // In parallel mode, check if kangaroos are on solved partitions
        if (parallel_mode) {
            for (int i = 0; i < BATCH_K; i++) {
                if (!needs_respawn[i] && partition_solved[kang_part[i]])
                    needs_respawn[i] = 1;
            }
        }

        int respawn_count = 0;
        for (int i = 0; i < BATCH_K; i++) {
            if (needs_respawn[i]) {
                respawn_count++;
                steps_since_respawn[i] = 1;
                visited_dp_cnt[i] = 0;

                // Pick partition: round-robin across all unsolved partitions
                if (num_partitions > 1) {
                    int p = pick_unsolved_partition(&rng, num_partitions);
                    if (p < 0) {
                        // All partitions solved — we're done
                        shutdown_flag = 1;
                        break;
                    }
                    kang_part[i] = p;
                }
                Point* my_wb = &partition_wild_base[kang_part[i]];

                if (pool_avail > 0 && num_partitions == 1) {
                    // Pool is only valid for single-partition mode
                    point_copy(&cx[i], &pool[pool_head].pt);
                    cd[i] = pool[pool_head].dist;
                    pool_head = (pool_head + 1) % POOL_SIZE;
                    pool_avail--;
                } else {
                    __int128 d = rand_gaussian_centered(w_half, &rng);
                    uint64_t k[4] = {0,0,0,0}; __int128 ad = (d >= 0) ? d : -d;
                    k[0] = (uint64_t)ad; k[1] = (uint64_t)(ad >> 64);
                    scalar_mult(&cx[i], k, &G);
                    if (d < 0) point_neg(&cx[i], &cx[i]);
                    Point tmp; point_copy(&tmp, &cx[i]); point_add(&cx[i], my_wb, &tmp);
                    if (y_is_greater_than_half(cx[i].y)) { aecc_sub(cx[i].y, pp, cx[i].y); d = -d; }
                    cd[i] = d;
                }
                life[i] = 1; hist_idx[i] = 0;
                memset(&HIST_PT(i, 0), 0, sizeof(Point) * rt_history_size);
                memset(&HIST_DIST(i, 0), 0, sizeof(__int128) * rt_history_size);
                jump_idx[i] = ((cx[i].x[0] * 0x9e3779b97f4a7c15ULL) >> rt_global_bits) & table_mask;
                generation[i]++;
                local_buf1_idx[i] = 0;
                memset(&LB1(i, 0), 0, rt_local_buf1_size * sizeof(uint64_t));
                last_canonical_x0[i] = 0; last_canonical_x1[i] = 0;
                escape_idx_arr[i] = 0; consecutive_escapes[i] = 0;
                needs_respawn[i] = 0;
            }
        }
        if (shutdown_flag) break;

        batch_point_add(cx, cd, jump_idx, BATCH_K, local_jt);

        for (int i = 0; i < BATCH_K; i++) {
            if (y_is_greater_than_half(cx[i].y)) { aecc_sub(cx[i].y, pp, cx[i].y); cd[i] = -cd[i]; }
            life[i]++;
            steps_since_respawn[i]++;

            if (life[i] >= (int)life_limit) {
                wstats[wid].count_life++;
                wstats[wid].life_at_respawn_sum += steps_since_respawn[i];
                wstats[wid].life_at_respawn_count++;
                __sync_fetch_and_add(&stat_life_respawn, 1);
                needs_respawn[i] = 1; continue;
            }

            uint64_t xv = cx[i].x[0];

            // History cycle detection
            int in_hist = 0; int found_hist_idx = -1;
            uint64_t curr_x0 = cx[i].x[0], curr_x1 = cx[i].x[1];
            for (uint32_t h = 0; h < rt_history_size; h++) {
                if (HIST_PT(i, h).x[0] == curr_x0 && HIST_PT(i, h).x[1] == curr_x1 && (curr_x0 || curr_x1)) {
                    in_hist = 1; found_hist_idx = h; break;
                }
            }

            if (in_hist) {
                __sync_fetch_and_add(&stat_history_cycles, 1);
                wstats[wid].escape_jumps++;
                if (consecutive_escapes[i] >= 16) {
                    wstats[wid].count_escape++;
                    wstats[wid].life_at_respawn_sum += steps_since_respawn[i];
                    wstats[wid].life_at_respawn_count++;
                    __sync_fetch_and_add(&stat_escape_respawn, 1);
                    needs_respawn[i] = 1; continue;
                }
                uint64_t min_x0 = curr_x0, min_x1 = curr_x1;
                __int128 canonical_dist = cd[i];
                Point canonical_pt; point_copy(&canonical_pt, &cx[i]);
                int idx = found_hist_idx;
                while (idx != hist_idx[i]) {
                    if (HIST_PT(i, idx).x[1] < min_x1 ||
                        (HIST_PT(i, idx).x[1] == min_x1 && HIST_PT(i, idx).x[0] < min_x0)) {
                        min_x0 = HIST_PT(i, idx).x[0]; min_x1 = HIST_PT(i, idx).x[1];
                        canonical_dist = HIST_DIST(i, idx);
                        point_copy(&canonical_pt, &HIST_PT(i, idx));
                    }
                    idx = (idx + 1) % rt_history_size;
                }
                if (min_x0 == last_canonical_x0[i] && min_x1 == last_canonical_x1[i]) {
                    escape_idx_arr[i]++;
                    __sync_fetch_and_add(&stat_history_same_cycle, 1);
                }
                last_canonical_x0[i] = min_x0; last_canonical_x1[i] = min_x1;
                int ei = (escape_idx_arr[i] + (min_x0 >> 40)) & esc_mask;
                point_add(&cx[i], &canonical_pt, &escape_table[ei].pt);
                cd[i] = canonical_dist + escape_table[ei].dist;
                if (y_is_greater_than_half(cx[i].y)) { aecc_sub(cx[i].y, pp, cx[i].y); cd[i] = -cd[i]; }
                consecutive_escapes[i]++;
                hist_idx[i] = 0;
                memset(&HIST_PT(i, 0), 0, sizeof(Point) * rt_history_size);
                memset(&HIST_DIST(i, 0), 0, sizeof(__int128) * rt_history_size);
                xv = cx[i].x[0];
            }

            if (!in_hist) consecutive_escapes[i] = 0;

            point_copy(&HIST_PT(i, hist_idx[i]), &cx[i]);
            HIST_DIST(i, hist_idx[i]) = cd[i];
            hist_idx[i] = (hist_idx[i] + 1) % rt_history_size;

            // Local buf1 cycle detection
            if ((xv & l_mask) == 0 && (xv & g_mask) != 0) {
                uint64_t fp = cx[i].x[0] * 0x9e3779b97f4a7c15ULL ^ cx[i].x[1] * 0xc6a4a7935bd1e995ULL;
                fp ^= fp >> 33;
                int in_buf1 = 0;
                for (uint32_t j = 0; j < rt_local_buf1_size; j++) {
                    if (LB1(i, j) == fp && fp != 0) { in_buf1 = 1; break; }
                }
                if (in_buf1) {
                    wstats[wid].count_buf1++;
                    wstats[wid].life_at_respawn_sum += steps_since_respawn[i];
                    wstats[wid].life_at_respawn_count++;
                    __sync_fetch_and_add(&stat_buf1_respawn, 1);
                    needs_respawn[i] = 1; continue;
                }
                LB1(i, local_buf1_idx[i]) = fp;
                local_buf1_idx[i] = (local_buf1_idx[i] + 1) % rt_local_buf1_size;
            }

            // DP check (wild)
            if ((xv & g_mask) == 0) {
                wstats[wid].dp_hits_total++;
                wstats[wid].wild_dp_checked++;

                uint64_t xhi_w = cx[i].x[1];
                int already_w = 0;
                for (int v = 0; v < visited_dp_cnt[i]; v++) {
                    if (visited_dp[i][v] == xhi_w) { already_w = 1; break; }
                }
                if (already_w) {
                    __sync_fetch_and_add(&stat_visited_dp_cycle, 1);
                    wstats[wid].life_at_respawn_sum += steps_since_respawn[i];
                    wstats[wid].life_at_respawn_count++;
                    needs_respawn[i] = 1; continue;
                }
                if (visited_dp_cnt[i] < VISITED_DP_CAP)
                    visited_dp[i][visited_dp_cnt[i]++] = xhi_w;

                pend_x_hi[pend_cnt] = xhi_w;
                pend_dist[pend_cnt] = cd[i];
                pend_part[pend_cnt] = kang_part[i];
                pend_cnt++;
            }

            jump_idx[i] = ((xv * 0x9e3779b97f4a7c15ULL) >> rt_global_bits) & table_mask;
        }


        steps += BATCH_K + respawn_count;
        wstats[wid].total_steps = steps;
        worker_steps[wid * 8] = steps;

        // Refill respawn pool (single-partition mode only)
        if (num_partitions == 1) {
            int to_refill = POOL_SIZE - pool_avail;
            if (to_refill > respawn_count + 2) to_refill = respawn_count + 2;
            for (int p = 0; p < to_refill; p++) {
                int slot = (pool_head + pool_avail) % POOL_SIZE;
                pool_generate_one(&pool[slot], &pool_rng, &partition_wild_base[pid]);
                pool_avail++;
            }
        }

        // Collision check against tame DB
        if (pend_cnt > 0) {
            for (int k = 0; k < pend_cnt; k++) {
                // In parallel mode, skip if this partition already solved
                if (parallel_mode && partition_solved[pend_part[k]])
                    continue;

                __int128 td[MAX_FP_MATCHES];
                int n_matches = 0;
                if (tame_db_lookup_hash(pend_x_hi[k], td, &n_matches)) {
                    wstats[wid].wild_dp_matched++;
                    __sync_fetch_and_add(&wild_db_matches, 1);
                    int solved = 0;
                    // td[m] has low rt_trunc_bits zeroed; try 2^T candidates per formula
                    int trunc_range = (rt_trunc_bits > 0) ? (1 << rt_trunc_bits) : 1;
                    Point neg_G; memcpy(neg_G.x, Gx, 32); memcpy(neg_G.y, Gy, 32);
                    point_neg(&neg_G, &neg_G);
                    const Point* target_pt = &partition_target[pend_part[k]];
                    for (int m = 0; m < n_matches && !solved; m++) {
                        for (int f = 0; f < 2 && !solved; f++) {
                            __int128 base_key;
                            if (f == 0) base_key =  td[m] + midpoint - pend_dist[k];
                            else        base_key = -td[m] + midpoint + pend_dist[k];
                            uint64_t kf[4] = {0,0,0,0};
                            __int128 ad = (base_key >= 0) ? base_key : -base_key;
                            kf[0] = (uint64_t)ad; kf[1] = (uint64_t)(ad >> 64);
                            Point chk; scalar_mult(&chk, kf, &G);
                            if (base_key < 0) point_neg(&chk, &chk);
                            const Point* step = (f == 0) ? &G : &neg_G;
                            for (int delta = 0; delta < trunc_range && !solved; delta++) {
                                if (delta > 0) {
                                    Point tmp; point_copy(&tmp, &chk);
                                    point_add(&chk, &tmp, step);
                                }
                                if (!memcmp(chk.x, target_pt->x, 32)) {
                                    solved = 1;
                                    __int128 real_key = (f == 0) ? base_key + delta
                                                                 : base_key - delta;
                                    __int128 ra = (real_key >= 0) ? real_key : -real_key;
                                    uint64_t rk[4] = {0,0,0,0};
                                    rk[0] = (uint64_t)ra; rk[1] = (uint64_t)(ra >> 64);
                                    pthread_mutex_lock(&db_mutex);
                                    if (parallel_mode) {
                                        if (!partition_solved[pend_part[k]]) {
                                            partition_solved[pend_part[k]] = 1;
                                            memcpy(partition_found_key[pend_part[k]], rk, 32);
                                            int remaining = __sync_sub_and_fetch((volatile int*)&num_unsolved, 1);
                                            printf("\n[!!!] KEY %d/%d FOUND (f%d m%d/%d delta=%d): 0x%lx%016lx [%d remaining]\n",
                                                   pend_part[k]+1, num_partitions, f+1, m+1, n_matches,
                                                   delta, rk[1], rk[0], remaining);
                                            if (remaining <= 0) {
                                                found_flag = 1;
                                                shutdown_flag = 1;
                                            }
                                        }
                                    } else {
                                        if (!found_flag) {
                                            found_flag = 1; found_partition = pend_part[k];
                                            memcpy(found_key, rk, 32);
                                            printf("\n[!!!] KEY FOUND (f%d m%d/%d delta=%d partition=%d): 0x%lx%016lx\n",
                                                   f+1, m+1, n_matches, delta, pend_part[k], rk[1], rk[0]);
                                        }
                                        shutdown_flag = 1;
                                    }
                                    pthread_mutex_unlock(&db_mutex);
                                }
                            }
                        }
                    }
                }
            }
            pend_cnt = 0;
        }
    }

    #undef LB1
    #undef HIST_PT
    #undef HIST_DIST
    worker_done[wid] = 1;
    free(local_jt); free(local_buf1_flat); free(generation); free(pool);
    free(hist_pt_flat); free(hist_dist_flat);
    return NULL;
}

// ==========================================
// SOLVE ONE KEY
// ==========================================

static int solve_one_key(const char* pubkey_hex, int key_index, int total_keys) {
    parse_pubkey(pubkey_hex, &target_point);
    uint64_t mk[4] = {(uint64_t)midpoint, (uint64_t)(midpoint >> 64), 0, 0};
    Point G, mp, nm; memcpy(G.x, Gx, 32); memcpy(G.y, Gy, 32);
    scalar_mult(&mp, mk, &G); point_neg(&nm, &mp);
    point_add(&wild_base, &target_point, &nm);

    // Set up partition 0 (normal mode: single partition)
    num_partitions = 1;
    parallel_mode = 0;
    point_copy(&partition_target[0], &target_point);
    point_copy(&partition_wild_base[0], &wild_base);
    partition_offset[0] = 0;

    shutdown_flag = 0; found_flag = 0; found_partition = 0;
    memset((void*)found_key, 0, 32);
    wild_db_matches = 0;
    memset((void*)worker_done, 0, sizeof(worker_done));
    memset((void*)worker_steps, 0, sizeof(worker_steps));
    memset(wstats, 0, sizeof(wstats));
    stat_history_cycles = 0; stat_history_same_cycle = 0;
    stat_escape_respawn = 0; stat_buf1_respawn = 0;
    stat_life_respawn = 0; stat_visited_dp_cycle = 0;

    printf("\n[KEY %d/%d] Searching: %s\n", key_index, total_keys, pubkey_hex);

    struct timespec ts, te; clock_gettime(CLOCK_MONOTONIC, &ts);
    pthread_t th[MAX_NUM_WORKERS]; WorkerData wd[MAX_NUM_WORKERS];
    for (int i = 0; i < NUM_WORKERS; i++) {
        wd[i].wid = i; wd[i].mode = MODE_WILD; wd[i].partition_id = 0;
        pthread_create(&th[i], NULL, worker_thread, &wd[i]);
    }

    while (!found_flag && !shutdown_flag) {
        usleep(100000);  // 100ms check interval
        struct timespec now; clock_gettime(CLOCK_MONOTONIC, &now);
        double el = (now.tv_sec - ts.tv_sec) + (now.tv_nsec - ts.tv_nsec) / 1e9;
        // Print status every ~3 seconds (only for slow keys)
        if ((int)el % 3 == 0 && el > 2.5) {
            uint64_t st = 0;
            for (int i = 0; i < NUM_WORKERS; i++) st += worker_steps[i*8];
            printf("[WILD %d/%d] %.1fM steps | %.1fM/s | %0.fs elapsed\n",
                   key_index, total_keys, st/1e6, st/el/1e6, el);
            usleep(900000);
        }
    }

    for (int i = 0; i < NUM_WORKERS; i++) pthread_join(th[i], NULL);

    clock_gettime(CLOCK_MONOTONIC, &te);
    double elapsed = (te.tv_sec - ts.tv_sec) + (te.tv_nsec - ts.tv_nsec) / 1e9;

    uint64_t total_steps = 0;
    for (int i = 0; i < NUM_WORKERS; i++) total_steps += worker_steps[i*8];

    if (found_flag) {
        printf("[SOLVED %d/%d] Key: 0x%lx%016lx | Steps: %.1fM | Time: %.3fs\n",
               key_index, total_keys, found_key[1], found_key[0], total_steps/1e6, elapsed);
        total_lookups = 0; disk_lookups = 0;
        return 1;
    } else {
        printf("[INTERRUPTED %d/%d] Search aborted after %.3fs\n", key_index, total_keys, elapsed);
        total_lookups = 0; disk_lookups = 0;
        return 0;
    }
}

// ==========================================
// SOLVE PARALLEL: multiple keys simultaneously
// ==========================================

static int solve_parallel(char** keys, int num_keys) {
    if (num_keys < 1) return 0;
    if (num_keys > MAX_PARTITIONS) {
        printf("[ERROR] Too many keys for parallel mode: %d (max %d)\n", num_keys, MAX_PARTITIONS);
        return 0;
    }

    printf("\n==========================================================\n");
    printf("   PARALLEL MODE: %d keys simultaneously\n", num_keys);
    printf("==========================================================\n");

    // Set up each key as a partition
    Point G_pt; memcpy(G_pt.x, Gx, 32); memcpy(G_pt.y, Gy, 32);
    uint64_t mk[4] = {(uint64_t)midpoint, (uint64_t)(midpoint >> 64), 0, 0};
    Point mp_G; scalar_mult(&mp_G, mk, &G_pt);
    Point neg_mp_G; point_neg(&neg_mp_G, &mp_G);

    num_partitions = num_keys;
    parallel_mode = 1;
    num_unsolved = num_keys;
    memset((void*)partition_solved, 0, sizeof(partition_solved));
    memset(partition_found_key, 0, sizeof(partition_found_key));

    for (int i = 0; i < num_keys; i++) {
        partition_pubkey_hex[i] = keys[i];
        parse_pubkey(keys[i], &partition_target[i]);
        point_add(&partition_wild_base[i], &partition_target[i], &neg_mp_G);
        partition_offset[i] = 0;
    }

    printf("[PARALLEL] Set up %d partitions (one per key)\n", num_keys);

    // Reset global state
    shutdown_flag = 0; found_flag = 0; found_partition = 0;
    memset((void*)found_key, 0, 32);
    wild_db_matches = 0;
    memset((void*)worker_done, 0, sizeof(worker_done));
    memset((void*)worker_steps, 0, sizeof(worker_steps));
    memset(wstats, 0, sizeof(wstats));
    stat_history_cycles = 0; stat_history_same_cycle = 0;
    stat_escape_respawn = 0; stat_buf1_respawn = 0;
    stat_life_respawn = 0; stat_visited_dp_cycle = 0;
    total_lookups = 0; disk_lookups = 0;

    struct timespec ts, te; clock_gettime(CLOCK_MONOTONIC, &ts);

    // Launch workers
    pthread_t th[MAX_NUM_WORKERS]; WorkerData wd[MAX_NUM_WORKERS];
    for (int i = 0; i < NUM_WORKERS; i++) {
        wd[i].wid = i; wd[i].mode = MODE_WILD; wd[i].partition_id = 0;
        pthread_create(&th[i], NULL, worker_thread, &wd[i]);
    }

    // Monitor loop
    int last_solved_count = 0;
    while (!shutdown_flag) {
        usleep(100000);  // 100ms
        if (sigint_received) { shutdown_flag = 1; break; }

        struct timespec now; clock_gettime(CLOCK_MONOTONIC, &now);
        double el = (now.tv_sec - ts.tv_sec) + (now.tv_nsec - ts.tv_nsec) / 1e9;

        int current_solved = 0;
        for (int i = 0; i < num_keys; i++)
            if (partition_solved[i]) current_solved++;

        // Check if all solved
        if (current_solved >= num_keys) {
            found_flag = 1;
            shutdown_flag = 1;
            break;
        }

        // Print status every ~3 seconds
        if ((int)el % 3 == 0 && el > 2.5) {
            uint64_t st = 0;
            for (int i = 0; i < NUM_WORKERS; i++) st += worker_steps[i*8];
            printf("[PARALLEL] %.1fM steps | %.1fM/s | %.0fs | solved: %d/%d\n",
                   st/1e6, st/el/1e6, el, current_solved, num_keys);
            usleep(900000);
        }
    }

    shutdown_flag = 1;
    for (int i = 0; i < NUM_WORKERS; i++) pthread_join(th[i], NULL);

    clock_gettime(CLOCK_MONOTONIC, &te);
    double elapsed = (te.tv_sec - ts.tv_sec) + (te.tv_nsec - ts.tv_nsec) / 1e9;

    uint64_t total_steps = 0;
    for (int i = 0; i < NUM_WORKERS; i++) total_steps += worker_steps[i*8];

    // Report results
    int solved_count = 0;
    printf("\n==========================================================\n");
    printf("   PARALLEL RESULTS\n");
    printf("==========================================================\n");
    for (int i = 0; i < num_keys; i++) {
        if (partition_solved[i]) {
            solved_count++;
            printf("[SOLVED %d/%d] Key: 0x%lx%016lx | %s\n",
                   i+1, num_keys,
                   partition_found_key[i][1], partition_found_key[i][0],
                   keys[i]);
        } else {
            printf("[UNSOLVED %d/%d] %s\n", i+1, num_keys, keys[i]);
        }
    }
    printf("----------------------------------------------------------\n");
    printf("   Solved: %d/%d | Steps: %.1fM | Time: %.3fs | %.1fM/s\n",
           solved_count, num_keys, total_steps/1e6, elapsed, total_steps/elapsed/1e6);
    if (solved_count > 0)
        printf("   Avg time/key: %.3fs | Avg steps/key: %.2fM\n",
               elapsed / solved_count, total_steps / (double)solved_count / 1e6);
    printf("==========================================================\n");

    print_instrumentation_report(NUM_WORKERS);

    // Reset parallel mode
    parallel_mode = 0;
    total_lookups = 0; disk_lookups = 0;
    return solved_count;
}

// ==========================================
// TEST MODE
// ==========================================
#define NUM_TEST_KEYS 2000

static void gen_key(__int128* k) {
    unsigned __int128 lo = (unsigned __int128)1 << rt_range_bits_low;
    unsigned __int128 hi = (unsigned __int128)1 << rt_range_bits_high;
    uint64_t r1 = ((uint64_t)rand()<<48)^((uint64_t)rand()<<32)^((uint64_t)rand()<<16)^rand();
    uint64_t r2 = ((uint64_t)rand()<<48)^((uint64_t)rand()<<32)^((uint64_t)rand()<<16)^rand();
    *k = (__int128)(lo + (((unsigned __int128)r2 << 64) | r1) % (hi - lo));
}

static void gen_key_ext(__int128* k, int ext_low, int ext_high) {
    unsigned __int128 lo = (unsigned __int128)1 << ext_low;
    unsigned __int128 hi = (unsigned __int128)1 << ext_high;
    uint64_t r1 = ((uint64_t)rand()<<48)^((uint64_t)rand()<<32)^((uint64_t)rand()<<16)^rand();
    uint64_t r2 = ((uint64_t)rand()<<48)^((uint64_t)rand()<<32)^((uint64_t)rand()<<16)^rand();
    *k = (__int128)(lo + (((unsigned __int128)r2 << 64) | r1) % (hi - lo));
}

static int cmp_dbl(const void* a, const void* b) { double d = *(double*)a - *(double*)b; return (d > 0) - (d < 0); }

static void run_tests(void) {
    printf("\n==========================================================\n");
    printf("   BENCHMARK (WILD-ONLY) - SCORED DP\n");
    printf("==========================================================\n");

    __int128 keys[NUM_TEST_KEYS]; double times[NUM_TEST_KEYS]; uint64_t steps_per_key[NUM_TEST_KEYS];
    srand(54321);
    for (int i = 0; i < NUM_TEST_KEYS; i++) gen_key(&keys[i]);

    Point G; memcpy(G.x, Gx, 32); memcpy(G.y, Gy, 32);
    double total = 0;
    uint64_t total_steps_all = 0;
    WorkerStats grand_total = {0};
    int num_completed = NUM_TEST_KEYS;

    for (int t = 0; t < NUM_TEST_KEYS; t++) {
        __int128 sk = keys[t];
        uint64_t kv[4] = {(uint64_t)sk, (uint64_t)(sk >> 64), 0, 0};
        scalar_mult(&target_point, kv, &G);
        uint64_t mk[4] = {(uint64_t)midpoint, (uint64_t)(midpoint >> 64), 0, 0};
        Point mp, nm; scalar_mult(&mp, mk, &G); point_neg(&nm, &mp);
        point_add(&wild_base, &target_point, &nm);

        num_partitions = 1;
        parallel_mode = 0;
        point_copy(&partition_target[0], &target_point);
        point_copy(&partition_wild_base[0], &wild_base);
        partition_offset[0] = 0;

        shutdown_flag = 0; found_flag = 0; found_partition = 0;
        memset((void*)found_key, 0, 32);
        wild_db_matches = 0;
        memset((void*)worker_done, 0, sizeof(worker_done));
        memset((void*)worker_steps, 0, sizeof(worker_steps));
        memset(wstats, 0, sizeof(wstats));
        stat_history_cycles = 0; stat_history_same_cycle = 0;
        stat_escape_respawn = 0; stat_buf1_respawn = 0;
        stat_life_respawn = 0; stat_visited_dp_cycle = 0;

        struct timespec ts, te; clock_gettime(CLOCK_MONOTONIC, &ts);
        pthread_t th[MAX_NUM_WORKERS]; WorkerData wd[MAX_NUM_WORKERS];
        for (int i = 0; i < NUM_WORKERS; i++) {
            wd[i].wid = i; wd[i].mode = MODE_WILD; wd[i].partition_id = 0;
            pthread_create(&th[i], NULL, worker_thread, &wd[i]);
        }
        while (!found_flag && !shutdown_flag) usleep(10000);
        shutdown_flag = 1;
        for (int i = 0; i < NUM_WORKERS; i++) pthread_join(th[i], NULL);

        if (!found_flag) {
            printf("\n[TEST] Interrupted after %d/%d keys\n", t, NUM_TEST_KEYS);
            num_completed = t;
            break;
        }
        clock_gettime(CLOCK_MONOTONIC, &te);

        double d = (te.tv_sec - ts.tv_sec) + (te.tv_nsec - ts.tv_nsec) / 1e9;
        times[t] = d; total += d;
        uint64_t tsteps = 0;
        for (int i = 0; i < NUM_WORKERS; i++) tsteps += worker_steps[i * 8];
        total_steps_all += tsteps;
        steps_per_key[t] = tsteps;

        for (int i = 0; i < NUM_WORKERS; i++) {
            grand_total.total_steps += wstats[i].total_steps;
            grand_total.count_life += wstats[i].count_life;
            grand_total.count_buf1 += wstats[i].count_buf1;
            grand_total.count_escape += wstats[i].count_escape;
            grand_total.escape_jumps += wstats[i].escape_jumps;
            grand_total.life_at_respawn_sum += wstats[i].life_at_respawn_sum;
            grand_total.life_at_respawn_count += wstats[i].life_at_respawn_count;
            grand_total.dp_hits_total += wstats[i].dp_hits_total;
            grand_total.dp_filtered += wstats[i].dp_filtered;
            grand_total.dp_deduped += wstats[i].dp_deduped;
            grand_total.dp_saved += wstats[i].dp_saved;
            grand_total.wild_dp_checked += wstats[i].wild_dp_checked;
            grand_total.wild_dp_matched += wstats[i].wild_dp_matched;
        }

        printf("[TEST %02d/%d] %.3fs | steps:%.1fM\n", t+1, NUM_TEST_KEYS, d, tsteps/1e6);
        total_lookups = 0; disk_lookups = 0;
    }

    if (num_completed == 0) { printf("\n[TEST] No keys completed.\n"); return; }

    qsort(times, num_completed, sizeof(double), cmp_dbl);
    double med = (num_completed % 2) ? times[num_completed/2] : (times[num_completed/2-1] + times[num_completed/2]) / 2;

    printf("\n==========================================================\n");
    printf("   RESULTS - SCORED DP BENCHMARK\n");
    printf("==========================================================\n");
    printf("   Min: %.3fs | Max: %.3fs | Mean: %.3fs\n", times[0], times[num_completed-1], total/num_completed);
    printf("   >>> MEDIAN: %.3fs <<<\n", med);
    printf("   Mean steps/key: %.2fM\n", total_steps_all / (double)num_completed / 1e6);
    double mean_steps = total_steps_all / (double)num_completed;
    double var_sum = 0;
    for (int i = 0; i < num_completed; i++) {
        double diff = (double)steps_per_key[i] - mean_steps;
        var_sum += diff * diff;
    }
    double stddev_steps = sqrt(var_sum / num_completed);
    double stderr_steps = stddev_steps / sqrt((double)num_completed);
    printf("   StdDev: %.2fM | StdErr mean: %.2fM (%.1f%%)\n",
           stddev_steps / 1e6, stderr_steps / 1e6, stderr_steps / mean_steps * 100.0);
    printf("==========================================================\n");

    printf("\n   AGGREGATE OVER %d KEYS:\n", num_completed);
    memset(wstats, 0, sizeof(wstats));
    wstats[0] = grand_total;
    print_instrumentation_report(1);

    // Theoretical analysis
    double actual_mean = grand_total.total_steps / (double)num_completed;
    long double N = (long double)tame_db_count;
    long double D = powl(2.0L, rt_global_bits);
    long double C = (long double)(NUM_WORKERS * BATCH_K);
    long double W = (powl(2.0L, rt_range_bits_high) - powl(2.0L, rt_range_bits_low)) / 2.0L;

    long double term1 = W / (N * D);
    long double term2 = 4.5L * C * D;
    long double theoretical_random = term1 + term2;

    double R_empirical = (actual_mean > (double)term2) ?
        (double)(term1 / (actual_mean - (double)term2)) : 0;
    if (R_empirical < 1.0) R_empirical = 1.0;

    double R_predicted = g_R_factor;
    long double theoretical_scored = (R_predicted > 0) ?
        term1 / R_predicted + term2 : theoretical_random;
    double deviation_pct = (R_predicted > 0) ?
        (actual_mean - (double)theoretical_scored) / (double)theoretical_scored * 100.0 : 0;
    double sigma = stderr_steps;
    double distance_sigma = (sigma > 0 && R_predicted > 0) ?
        (actual_mean - (double)theoretical_scored) / sigma : 0;

    printf("\n==========================================================\n");
    printf("   THEORETICAL ANALYSIS\n");
    printf("==========================================================\n");
    printf("   Range:           2^%u - 2^%u\n", rt_range_bits_low, rt_range_bits_high);
    printf("   N (DB tame):     %.2fK\n", tame_db_count / 1e3);
    printf("   D (DP mask):     2^%u\n", rt_global_bits);
    printf("   C (parallelism): %d x %d = %d\n", NUM_WORKERS, BATCH_K, NUM_WORKERS * BATCH_K);
    printf("   Latency:         4.5*C*D = %.2fM\n", (double)term2 / 1e6);
    printf("   ---\n");
    printf("   Random DP est:   %.2fM steps/key\n", (double)theoretical_random / 1e6);
    if (R_predicted > 0) {
        printf("   R predicted:     %.1f (q_hat=%.6f)\n", R_predicted, g_q_hat);
        printf("   Scored est:      %.2fM steps/key\n", (double)theoretical_scored / 1e6);
    }
    printf("   R empirical:     %.1f\n", R_empirical);
    printf("   Actual mean:     %.2fM steps/key\n", actual_mean / 1e6);
    if (R_predicted > 0) {
        printf("   Deviation:       %+.2f%%\n", deviation_pct);
        printf("   Distance:        %.2f sigma\n", distance_sigma);
    }
    printf("==========================================================\n");
}

// ==========================================
// GEN_TEST MODE
// ==========================================

static void run_gen_test(int num_keys) {
    printf("[GEN_TEST] Generating %d random public keys...\n", num_keys);
    srand(54321);
    Point G; memcpy(G.x, Gx, 32); memcpy(G.y, Gy, 32);

    FILE* f = fopen(PUBKEY_LIST_FILENAME, "w");
    if (!f) { printf("[ERROR] Cannot open %s for writing\n", PUBKEY_LIST_FILENAME); return; }

    for (int i = 0; i < num_keys; i++) {
        __int128 sk; gen_key(&sk);
        uint64_t kv[4] = {(uint64_t)sk, (uint64_t)(sk >> 64), 0, 0};
        Point pub; scalar_mult(&pub, kv, &G);
        char hex[67];
        int prefix = (pub.y[0] & 1) ? 0x03 : 0x02;
        sprintf(hex, "%02x", prefix);
        for (int j = 0; j < 4; j++) sprintf(hex + 2 + j*16, "%016lx", pub.x[3-j]);
        fprintf(f, "%s\n", hex);
    }
    fclose(f);
    printf("[GEN_TEST] Written %d keys to %s\n", num_keys, PUBKEY_LIST_FILENAME);
}

static void run_gen_test_ext(int num_keys, int ext_low, int ext_high) {
    printf("[GEN_TEST_EXT] Generating %d random public keys in range 2^%d - 2^%d ...\n",
           num_keys, ext_low, ext_high);
    srand(54321);
    Point G; memcpy(G.x, Gx, 32); memcpy(G.y, Gy, 32);

    FILE* f = fopen(PUBKEY_LIST_FILENAME, "w");
    if (!f) { printf("[ERROR] Cannot open %s for writing\n", PUBKEY_LIST_FILENAME); return; }

    for (int i = 0; i < num_keys; i++) {
        __int128 sk; gen_key_ext(&sk, ext_low, ext_high);
        uint64_t kv[4] = {(uint64_t)sk, (uint64_t)(sk >> 64), 0, 0};
        Point pub; scalar_mult(&pub, kv, &G);
        char hex[67];
        int prefix = (pub.y[0] & 1) ? 0x03 : 0x02;
        sprintf(hex, "%02x", prefix);
        for (int j = 0; j < 4; j++) sprintf(hex + 2 + j*16, "%016lx", pub.x[3-j]);
        fprintf(f, "%s\n", hex);
    }
    fclose(f);
    printf("[GEN_TEST_EXT] Written %d keys to %s\n", num_keys, PUBKEY_LIST_FILENAME);
}

// ==========================================
// LOAD KEY LIST
// ==========================================

static int load_pubkey_list(const char* filename, char*** keys_out) {
    FILE* f = fopen(filename, "r");
    if (!f) { printf("[ERROR] Cannot open %s\n", filename); return 0; }
    int cap = 256, count = 0;
    char** keys = malloc(cap * sizeof(char*));
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        int len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
        if (len < 66) continue;
        if (line[0] != '0' || (line[1] != '2' && line[1] != '3')) continue;
        if (count >= cap) { cap *= 2; keys = realloc(keys, cap * sizeof(char*)); }
        keys[count] = strdup(line);
        count++;
    }
    fclose(f);
    *keys_out = keys;
    return count;
}

// ==========================================
// SOLVE ONE KEY - EXTEND MODE
// ==========================================

static int solve_one_key_extend(const char* pubkey_hex, int ext_low, int ext_high,
                                uint64_t npart, unsigned __int128 db_range,
                                int key_index, int total_keys) {
    Point Q;
    parse_pubkey(pubkey_hex, &Q);
    Point G_pt; memcpy(G_pt.x, Gx, 32); memcpy(G_pt.y, Gy, 32);

    unsigned __int128 ext_low_val = (unsigned __int128)1 << ext_low;
    unsigned __int128 db_low_val  = (unsigned __int128)1 << rt_range_bits_low;
    unsigned __int128 base_offset = ext_low_val - db_low_val;

    // Precompute base_offset * G
    uint64_t bo_k[4] = {(uint64_t)base_offset, (uint64_t)(base_offset >> 64), 0, 0};
    Point base_offset_G; scalar_mult(&base_offset_G, bo_k, &G_pt);
    Point neg_base_offset_G; point_neg(&neg_base_offset_G, &base_offset_G);

    // Precompute db_range * G
    uint64_t dr_k[4] = {(uint64_t)db_range, (uint64_t)(db_range >> 64), 0, 0};
    Point db_range_G; scalar_mult(&db_range_G, dr_k, &G_pt);
    Point neg_db_range_G; point_neg(&neg_db_range_G, &db_range_G);

    // Precompute midpoint * G (for wild_base)
    uint64_t mk[4] = {(uint64_t)midpoint, (uint64_t)(midpoint >> 64), 0, 0};
    Point mp_G; scalar_mult(&mp_G, mk, &G_pt);
    Point neg_mp_G; point_neg(&neg_mp_G, &mp_G);

    num_partitions = (int)npart;
    parallel_mode = 0;
    for (int k = 0; k < (int)npart; k++) {
        if (k == 0)
            point_add(&partition_target[0], &Q, &neg_base_offset_G);
        else
            point_add(&partition_target[k], &partition_target[k-1], &neg_db_range_G);
        point_add(&partition_wild_base[k], &partition_target[k], &neg_mp_G);
        partition_offset[k] = base_offset + (unsigned __int128)k * db_range;
    }

    rt_life_limit = VITA_WILD_MAX;

    shutdown_flag = 0; found_flag = 0; found_partition = 0;
    memset((void*)found_key, 0, 32);
    wild_db_matches = 0;
    memset((void*)worker_done, 0, sizeof(worker_done));
    memset((void*)worker_steps, 0, sizeof(worker_steps));
    memset(wstats, 0, sizeof(wstats));
    stat_history_cycles = 0; stat_history_same_cycle = 0;
    stat_escape_respawn = 0; stat_buf1_respawn = 0;
    stat_life_respawn = 0; stat_visited_dp_cycle = 0;

    printf("\n[KEY %d/%d] Searching (extend 2^%d-2^%d, %lu parts): %s\n",
           key_index, total_keys, ext_low, ext_high, (unsigned long)npart, pubkey_hex);

    struct timespec ts, te; clock_gettime(CLOCK_MONOTONIC, &ts);
    pthread_t th[MAX_NUM_WORKERS]; WorkerData wd[MAX_NUM_WORKERS];
    for (int i = 0; i < NUM_WORKERS; i++) {
        wd[i].wid = i; wd[i].mode = MODE_WILD; wd[i].partition_id = 0;
        pthread_create(&th[i], NULL, worker_thread, &wd[i]);
    }

    while (!found_flag && !shutdown_flag) {
        usleep(100000);
        struct timespec now; clock_gettime(CLOCK_MONOTONIC, &now);
        double el = (now.tv_sec - ts.tv_sec) + (now.tv_nsec - ts.tv_nsec) / 1e9;
        if ((int)el % 3 == 0 && el > 2.5) {
            uint64_t st = 0;
            for (int i = 0; i < NUM_WORKERS; i++) st += worker_steps[i*8];
            printf("[EXTEND %d/%d] %.1fM steps | %.1fM/s | %.0fs elapsed\n",
                   key_index, total_keys, st/1e6, st/el/1e6, el);
            usleep(900000);
        }
    }

    shutdown_flag = 1;
    for (int i = 0; i < NUM_WORKERS; i++) pthread_join(th[i], NULL);

    clock_gettime(CLOCK_MONOTONIC, &te);
    double elapsed = (te.tv_sec - ts.tv_sec) + (te.tv_nsec - ts.tv_nsec) / 1e9;

    uint64_t total_steps = 0;
    for (int i = 0; i < NUM_WORKERS; i++) total_steps += worker_steps[i*8];

    if (found_flag) {
        unsigned __int128 real_key = 0;
        real_key = (unsigned __int128)found_key[0] | ((unsigned __int128)found_key[1] << 64);
        real_key += partition_offset[found_partition];
        uint64_t rk[2] = {(uint64_t)real_key, (uint64_t)(real_key >> 64)};

        printf("[SOLVED %d/%d] Key: 0x%lx%016lx (part %d) | Steps: %.1fM | Time: %.3fs\n",
               key_index, total_keys, rk[1], rk[0], found_partition, total_steps/1e6, elapsed);

        // Verify
        Point chk;
        uint64_t vk[4] = {rk[0], rk[1], 0, 0};
        scalar_mult(&chk, vk, &G_pt);
        if (memcmp(chk.x, Q.x, 32))
            printf("[VERIFY] MISMATCH for key %d/%d!\n", key_index, total_keys);

        // Store real key in found_key for caller
        found_key[0] = rk[0]; found_key[1] = rk[1];
        total_lookups = 0; disk_lookups = 0;
        return 1;
    } else {
        printf("[INTERRUPTED %d/%d] Search aborted after %.3fs\n", key_index, total_keys, elapsed);
        total_lookups = 0; disk_lookups = 0;
        return 0;
    }
}

// ==========================================
// SIGNAL HANDLER
// ==========================================

static void sigint_handler(int sig) {
    (void)sig;
    printf("\n[SIGINT] Ctrl+C received, shutting down...\n");
    sigint_received = 1;
    shutdown_flag = 1;
}

// ==========================================
// MAIN
// ==========================================

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    signal(SIGINT, sigint_handler);

    NUM_WORKERS = DEFAULT_NUM_WORKERS;
    BATCH_K = DEFAULT_BATCH_K;

    // ======================================
    // HELP
    // ======================================
    if (argc > 1 && (!strcasecmp(argv[1], "help") || !strcasecmp(argv[1], "-h") || !strcasecmp(argv[1], "--help"))) {
        printf("============================================================\n");
        printf("  KANGAROO WILD - Pollard's Kangaroo Wild Search\n");
        printf("============================================================\n\n");
        printf("Usage: %s [command] [args]\n\n", argv[0]);
        printf("Commands:\n\n");
        printf("  (default)   Load public keys from %s and solve each one.\n", PUBKEY_LIST_FILENAME);
        printf("              Requires the tame DB files in the current directory.\n");
        printf("              Optional: %s <file> <range_low> <range_high>\n", argv[0]);
        printf("              If range is larger than DB, automatically uses extend mode.\n\n");
        printf("  parallel [file]\n");
        printf("              Load public keys from file (default: %s) and\n", PUBKEY_LIST_FILENAME);
        printf("              search for ALL of them simultaneously. Each key becomes\n");
        printf("              a partition; workers distribute kangaroos across all keys.\n");
        printf("              Keys must be in the same range as the DB (2^%u - 2^%u).\n",
               0, 0);  // will show 0-0 in help, real values shown at runtime
        printf("              Solved keys are removed from the search automatically.\n");
        printf("              Max keys: %d\n\n", MAX_PARTITIONS);
        printf("  test        Generate %d random keys and benchmark solving them.\n", NUM_TEST_KEYS);
        printf("              Reports timing statistics and theoretical comparison.\n\n");
        printf("  gen_test [N]  Generate N random public keys (default %d) and save\n", NUM_TEST_KEYS);
        printf("              them to %s. Does NOT solve them.\n\n", PUBKEY_LIST_FILENAME);
        printf("  gen_test_ext <N> <range_low> <range_high>\n");
        printf("              Generate N random public keys in an extended range.\n");
        printf("              Example: %s gen_test_ext 100 69 70\n\n", argv[0]);
        printf("  single <pubkey>\n");
        printf("              Solve a single compressed public key provided on the\n");
        printf("              command line. Example:\n");
        printf("              %s single 035d039b....\n\n", argv[0]);
        printf("  extend <pubkey> <range_low> <range_high>\n");
        printf("              Solve a key whose private key lies in a LARGER range\n");
        printf("              than the DB covers. The program partitions the extended\n");
        printf("              range and searches all partitions in parallel.\n");
        printf("              Example: %s extend 03abcd... 66 67\n\n", argv[0]);
        printf("  help        Show this help message.\n\n");
        printf("Required DB files (generated by tame phase):\n");
        printf("  %s\n", DB_FILENAME);
        printf("  %s\n", FINGERPRINT_FILENAME);
        printf("  %s\n", BUCKET_OFFSETS_FILENAME);
        printf("  %s\n\n", TRAINING_PARAMS_FILENAME);
        printf("All internal parameters (range, DP bits, jump table, cycle detection,\n");
        printf("etc.) are auto-detected from the training_params file and DB files.\n\n");
        printf("Current settings:\n");
        printf("  NUM_WORKERS:    %d (worker threads)\n", NUM_WORKERS);
        printf("  BATCH_K:        %d (kangaroos per worker)\n", BATCH_K);
        printf("  C (parallelism): %d\n", NUM_WORKERS * BATCH_K);
        printf("  VITA_WILD_MAX:  %llu (max steps before respawn)\n", (unsigned long long)VITA_WILD_MAX);
        printf("============================================================\n");
        return 0;
    }

    // ======================================
    // GEN_TEST_EXT MODE: generate keys in extended range
    // Usage: gen_test_ext <N> <range_low> <range_high>
    // ======================================
    if (argc > 1 && !strcasecmp(argv[1], "gen_test_ext")) {
        aecc_init();
        int tp_ok = load_training_params();
        if (tp_ok <= 0) {
            printf("[ERROR] training_params file is required.\n");
            return 1;
        }
        if (argc < 5) {
            printf("Usage: %s gen_test_ext <N> <range_low> <range_high>\n", argv[0]);
            printf("Example: %s gen_test_ext 100 69 70\n", argv[0]);
            return 1;
        }
        int num = atoi(argv[2]);
        int ext_low = atoi(argv[3]);
        int ext_high = atoi(argv[4]);
        if (num < 1) num = 1;
        run_gen_test_ext(num, ext_low, ext_high);
        return 0;
    }

    // ======================================
    // GEN_TEST MODE
    // ======================================
    if (argc > 1 && !strcasecmp(argv[1], "gen_test")) {
        aecc_init();
        int tp_ok = load_training_params();
        if (tp_ok <= 0) {
            printf("[ERROR] training_params file is required.\n");
            return 1;
        }
        unsigned __int128 lo = (unsigned __int128)1 << rt_range_bits_low;
        unsigned __int128 hi = (unsigned __int128)1 << rt_range_bits_high;
        w_half = (__int128)((hi - lo) / 2);
        midpoint = (__int128)(lo + (unsigned __int128)w_half);

        int num = NUM_TEST_KEYS;
        if (argc > 2) num = atoi(argv[2]);
        if (num < 1) num = 1;
        run_gen_test(num);
        return 0;
    }

    // ======================================
    // All other modes: full initialization
    // ======================================
    aecc_init();
    int tp_ok = load_training_params();
    if (tp_ok <= 0) {
        printf("[ERROR] training_params file is required. Cannot proceed.\n");
        return 1;
    }

    unsigned __int128 lo = (unsigned __int128)1 << rt_range_bits_low;
    unsigned __int128 hi = (unsigned __int128)1 << rt_range_bits_high;
    w_quarto = (__int128)(hi - lo);
    w_half   = (__int128)((hi - lo) / 2);
    midpoint = (__int128)(lo + (unsigned __int128)w_half);

    init_jump_table();
    load_tame_db(DB_FILENAME);

    printf("--- KANGAROO WILD ---\n");
    printf("[CONFIG] Range: 2^%u - 2^%u | GLOBAL_BITS: %u | DIST_BYTES: %u | TRUNC_BITS: %u\n",
           rt_range_bits_low, rt_range_bits_high, rt_global_bits, rt_dist_bytes, rt_trunc_bits);
    printf("[CONFIG] Workers: %d | Batch: %d | C: %d | VITA: %llu\n",
           NUM_WORKERS, BATCH_K, NUM_WORKERS * BATCH_K, (unsigned long long)VITA_WILD_MAX);

    // ======================================
    // TEST MODE
    // ======================================
    if (argc > 1 && !strcasecmp(argv[1], "test")) {
        run_tests();
        goto cleanup;
    }

    // ======================================
    // SINGLE MODE
    // ======================================
    if (argc > 1 && !strcasecmp(argv[1], "single")) {
        if (argc < 3) {
            printf("Usage: %s single <compressed_pubkey_hex>\n", argv[0]);
            goto cleanup;
        }
        char* pubkey = argv[2];
        if (strlen(pubkey) < 66 || pubkey[0] != '0' || (pubkey[1] != '2' && pubkey[1] != '3')) {
            printf("[ERROR] Invalid compressed public key: %s\n", pubkey);
            printf("        Must be 66 hex chars starting with 02 or 03\n");
            goto cleanup;
        }
        solve_one_key(pubkey, 1, 1);
        print_instrumentation_report(NUM_WORKERS);
        if (found_flag) printf("\n[!!!] PRIVATE KEY: 0x%lx%016lx\n", found_key[1], found_key[0]);
        goto cleanup;
    }

    // ======================================
    // PARALLEL MODE: solve multiple keys simultaneously
    // Usage: parallel [file]
    // ======================================
    if (argc > 1 && !strcasecmp(argv[1], "parallel")) {
        const char* keyfile = PUBKEY_LIST_FILENAME;
        if (argc > 2) keyfile = argv[2];

        char** keys = NULL;
        int num_keys = load_pubkey_list(keyfile, &keys);
        if (num_keys == 0) {
            printf("[ERROR] No valid public keys found in %s\n", keyfile);
            goto cleanup;
        }
        printf("[PARALLEL] Loaded %d public keys from %s\n", num_keys, keyfile);
        printf("[PARALLEL] DB range: 2^%u - 2^%u | All keys searched simultaneously\n",
               rt_range_bits_low, rt_range_bits_high);

        int solved = solve_parallel(keys, num_keys);

        // Print all found private keys
        if (solved > 0) {
            printf("\n[!!!] FOUND PRIVATE KEYS:\n");
            for (int i = 0; i < num_keys; i++) {
                if (partition_solved[i])
                    printf("  [%d] 0x%lx%016lx  %s\n",
                           i+1, partition_found_key[i][1], partition_found_key[i][0], keys[i]);
            }
        }

        for (int i = 0; i < num_keys; i++) free(keys[i]);
        free(keys);
        goto cleanup;
    }

    // ======================================
    // EXTEND MODE: solve key in a larger range
    // Usage: extend <pubkey> <range_low> <range_high>
    // Example: extend 03abcd... 66 67
    //   DB covers 2^63-2^64, user wants 2^66-2^67
    //   => 2^(67-64) = 8 partitions, searched in parallel
    // ======================================
    if (argc > 1 && !strcasecmp(argv[1], "extend")) {
        if (argc < 5) {
            printf("Usage: %s extend <compressed_pubkey_hex> <range_low> <range_high>\n", argv[0]);
            printf("Example: %s extend 035d039b... 66 67\n", argv[0]);
            printf("  The DB covers 2^%u - 2^%u. Specify the actual range of the key.\n",
                   rt_range_bits_low, rt_range_bits_high);
            goto cleanup;
        }
        char* pubkey = argv[2];
        int ext_low = atoi(argv[3]);
        int ext_high = atoi(argv[4]);

        if (strlen(pubkey) < 66 || pubkey[0] != '0' || (pubkey[1] != '2' && pubkey[1] != '3')) {
            printf("[ERROR] Invalid compressed public key: %s\n", pubkey);
            goto cleanup;
        }

        int db_width = rt_range_bits_high - rt_range_bits_low;
        int ext_width = ext_high - ext_low;
        if (ext_width < db_width) {
            printf("[ERROR] Extended range (%d bits) smaller than DB range (%d bits)\n", ext_width, db_width);
            goto cleanup;
        }
        if (ext_width == db_width && ext_low == (int)rt_range_bits_low) {
            printf("[INFO] Extended range matches DB range — using normal single mode.\n");
            solve_one_key(pubkey, 1, 1);
            print_instrumentation_report(NUM_WORKERS);
            if (found_flag) printf("\n[!!!] PRIVATE KEY: 0x%lx%016lx\n", found_key[1], found_key[0]);
            goto cleanup;
        }

        unsigned __int128 db_range = ((unsigned __int128)1 << rt_range_bits_high)
                                   - ((unsigned __int128)1 << rt_range_bits_low);
        unsigned __int128 ext_range = ((unsigned __int128)1 << ext_high)
                                    - ((unsigned __int128)1 << ext_low);
        uint64_t npart = (uint64_t)(ext_range / db_range);
        if (npart < 1) npart = 1;
        if (npart > MAX_PARTITIONS) {
            printf("[ERROR] Too many partitions: %lu (max %d)\n",
                   (unsigned long)npart, MAX_PARTITIONS);
            goto cleanup;
        }

        printf("\n[EXTEND] Range 2^%d - 2^%d | DB 2^%u - 2^%u | %lu partitions\n",
               ext_low, ext_high, rt_range_bits_low, rt_range_bits_high, (unsigned long)npart);

        solve_one_key_extend(pubkey, ext_low, ext_high, npart, db_range, 1, 1);
        print_instrumentation_report(NUM_WORKERS);
        if (found_flag) printf("\n[!!!] PRIVATE KEY: 0x%lx%016lx\n", found_key[1], found_key[0]);
        goto cleanup;
    }

    // ======================================
    // DEFAULT MODE: load key list and solve
    // Usage: [file] [range_low range_high]
    // If range is specified and differs from DB, uses extend mode
    // ======================================
    {
        char** keys = NULL;
        const char* keyfile = PUBKEY_LIST_FILENAME;
        if (argc > 1) keyfile = argv[1];

        // Optional extended range
        int use_extend = 0;
        int ext_low = 0, ext_high = 0;
        uint64_t npart = 1;
        unsigned __int128 db_range = ((unsigned __int128)1 << rt_range_bits_high)
                                   - ((unsigned __int128)1 << rt_range_bits_low);

        if (argc > 3) {
            ext_low = atoi(argv[2]);
            ext_high = atoi(argv[3]);
            int db_width = rt_range_bits_high - rt_range_bits_low;
            int ext_width = ext_high - ext_low;
            if (ext_width >= db_width && !(ext_width == db_width && ext_low == (int)rt_range_bits_low)) {
                unsigned __int128 ext_range = ((unsigned __int128)1 << ext_high)
                                            - ((unsigned __int128)1 << ext_low);
                npart = (uint64_t)(ext_range / db_range);
                if (npart < 1) npart = 1;
                if (npart > MAX_PARTITIONS) {
                    printf("[ERROR] Too many partitions: %lu (max %d)\n",
                           (unsigned long)npart, MAX_PARTITIONS);
                    goto cleanup;
                }
                use_extend = 1;
                printf("[EXTEND] Range 2^%d - 2^%d | %lu partitions\n",
                       ext_low, ext_high, (unsigned long)npart);
            }
        }

        int num_keys = load_pubkey_list(keyfile, &keys);
        if (num_keys == 0) {
            printf("[ERROR] No valid public keys found in %s\n", keyfile);
            goto cleanup;
        }
        printf("[KEYS] Loaded %d public keys from %s\n", num_keys, keyfile);

        int solved = 0, failed = 0;
        double total_time = 0;
        uint64_t total_steps_all = 0;
        double* key_times = calloc(num_keys, sizeof(double));
        uint64_t* key_steps = calloc(num_keys, sizeof(uint64_t));

        for (int i = 0; i < num_keys; i++) {
            if (sigint_received) break;
            struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
            int ok;
            if (use_extend)
                ok = solve_one_key_extend(keys[i], ext_low, ext_high, npart, db_range, i+1, num_keys);
            else
                ok = solve_one_key(keys[i], i+1, num_keys);
            struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
            double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
            total_time += elapsed;
            uint64_t ksteps = 0;
            for (int w = 0; w < NUM_WORKERS; w++) ksteps += worker_steps[w*8];
            total_steps_all += ksteps;
            if (ok) {
                key_times[solved] = elapsed;
                key_steps[solved] = ksteps;
                solved++;
            } else { failed++; if (sigint_received) break; }
        }

        printf("\n==========================================================\n");
        printf("   SUMMARY\n");
        printf("==========================================================\n");
        printf("   Total keys:     %d\n", num_keys);
        printf("   Solved:         %d\n", solved);
        printf("   Failed/Aborted: %d\n", failed);
        if (solved > 0) {
            // Sort times for median/min/max
            for (int i = 0; i < solved - 1; i++)
                for (int j = i + 1; j < solved; j++)
                    if (key_times[i] > key_times[j]) {
                        double t = key_times[i]; key_times[i] = key_times[j]; key_times[j] = t;
                        uint64_t s = key_steps[i]; key_steps[i] = key_steps[j]; key_steps[j] = s;
                    }
            double median_t = (solved % 2) ? key_times[solved/2]
                : (key_times[solved/2-1] + key_times[solved/2]) / 2.0;
            double median_s = (solved % 2) ? (double)key_steps[solved/2]
                : (double)(key_steps[solved/2-1] + key_steps[solved/2]) / 2.0;

            printf("   Total time:     %.1fs\n", total_time);
            printf("   Min:            %.3fs | Max: %.3fs\n", key_times[0], key_times[solved-1]);
            printf("   Mean time/key:  %.3fs\n", total_time / solved);
            printf("   >>> MEDIAN:     %.3fs <<<\n", median_t);
            printf("   Mean steps/key: %.2fM\n", total_steps_all / (double)solved / 1e6);
            printf("   Median steps:   %.2fM\n", median_s / 1e6);

            // 95% confidence interval for the mean (steps)
            if (solved >= 2) {
                double mean_s = total_steps_all / (double)solved;
                double var_sum = 0;
                // key_steps is sorted, but variance doesn't depend on order
                // we need the original steps — re-derive from sorted array is fine
                for (int i = 0; i < solved; i++) {
                    double diff = (double)key_steps[i] - mean_s;
                    var_sum += diff * diff;
                }
                double stddev = sqrt(var_sum / (solved - 1));
                double stderr_mean = stddev / sqrt((double)solved);
                double ci_lo = mean_s - 1.96 * stderr_mean;
                double ci_hi = mean_s + 1.96 * stderr_mean;
                if (ci_lo < 0) ci_lo = 0;
                printf("   StdDev:         %.2fM\n", stddev / 1e6);
                printf("   95%% CI (mean):  [%.2fM, %.2fM] steps/key\n", ci_lo / 1e6, ci_hi / 1e6);
            }
        }
        printf("==========================================================\n");

        free(key_times); free(key_steps);
        for (int i = 0; i < num_keys; i++) free(keys[i]);
        free(keys);
    }

cleanup:
    if (fingerprints && fingerprints != MAP_FAILED)
        munmap(fingerprints, fingerprint_count * sizeof(uint32_t));
    if (bucket_offsets && bucket_offsets != MAP_FAILED)
        munmap(bucket_offsets, bucket_offsets_map_size);
    if (compact_db_array) {
        if (rt_dist_bytes == MAX_DIST_BYTES && compact_db_array != MAP_FAILED)
            munmap(compact_db_array, tame_db_file_size);
        else
            free(compact_db_array);
    }
    if (tame_db_fd != -1) close(tame_db_fd);
    free(jump_table); free(escape_table);
    return 0;
}
