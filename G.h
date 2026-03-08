#ifndef G_H
#define G_H

#include <stdint.h>

// Constants
extern const uint64_t pp[4];
extern const uint64_t b_powG[512][4];
extern const uint64_t b_G[4098][4];

// Field operations
uint8_t aecc_zero_p(uint64_t *xp);
int aecc_cmp(const uint64_t *ap, const uint64_t *bp);
void aecc_selfadd(uint64_t *rp, uint64_t b);
void aecc_sub(uint64_t *r, const uint64_t *a, const uint64_t *b);
void aecc_mul(uint64_t *rp, const uint64_t *up, const uint64_t *vp);
void aecc_sqr(uint64_t *rp, const uint64_t *up);
void aecc_inv(uint64_t *r, const uint64_t *a);

// Point operations
void aecc_add_ja(uint64_t *jaxp, uint64_t *jayp, uint64_t *jazp,
                 const uint64_t *jbxr, const uint64_t *jbyr);
void mul(uint64_t *kxp, uint64_t *kyp, uint64_t *kzp, const uint64_t *k);
void jac_to_aff(uint64_t *axp, uint64_t *ayp, 
                uint64_t *jaxr, uint64_t *jayr, uint64_t *invjazr);

// Init function (stub - not needed with G_stub.c)
static inline void aecc_init(void) {}

#endif // G_H
