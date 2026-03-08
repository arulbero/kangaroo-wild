// fast_inv.h - DRS62 modular inversion, simplified
// Based on code by Jean Luc Pons
// Original: https://github.com/JeanLucPons/Kangaroo
// Licensed under GPL-3.0

#ifndef FAST_INV_H
#define FAST_INV_H

#include <stdint.h>

static const uint64_t FI_P[5] = {
    0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0
};

#define FI_MM64 0xD838091DD2253531ULL
#define FI_MSK62 0x3FFFFFFFFFFFFFFFULL

#define UMUL(r,a,b) do { \
    __uint128_t _c = (__uint128_t)(a)[0]*(b); (r)[0]=(uint64_t)_c; _c>>=64; \
    _c += (__uint128_t)(a)[1]*(b); (r)[1]=(uint64_t)_c; _c>>=64; \
    _c += (__uint128_t)(a)[2]*(b); (r)[2]=(uint64_t)_c; _c>>=64; \
    _c += (__uint128_t)(a)[3]*(b); (r)[3]=(uint64_t)_c; _c>>=64; \
    _c += (__uint128_t)(a)[4]*(b); (r)[4]=(uint64_t)_c; } while(0)

#define ADD5(r,a) do { \
    __uint128_t _c = (__uint128_t)(r)[0]+(a)[0]; (r)[0]=(uint64_t)_c; _c>>=64; \
    _c += (__uint128_t)(r)[1]+(a)[1]; (r)[1]=(uint64_t)_c; _c>>=64; \
    _c += (__uint128_t)(r)[2]+(a)[2]; (r)[2]=(uint64_t)_c; _c>>=64; \
    _c += (__uint128_t)(r)[3]+(a)[3]; (r)[3]=(uint64_t)_c; _c>>=64; \
    _c += (__uint128_t)(r)[4]+(a)[4]; (r)[4]=(uint64_t)_c; } while(0)

#define SUB5(r,a) do { \
    __uint128_t _b=0,_d; \
    _d=(__uint128_t)(r)[0]-(a)[0]; (r)[0]=(uint64_t)_d; _b=_d>>127; \
    _d=(__uint128_t)(r)[1]-(a)[1]-_b; (r)[1]=(uint64_t)_d; _b=_d>>127; \
    _d=(__uint128_t)(r)[2]-(a)[2]-_b; (r)[2]=(uint64_t)_d; _b=_d>>127; \
    _d=(__uint128_t)(r)[3]-(a)[3]-_b; (r)[3]=(uint64_t)_d; _b=_d>>127; \
    _d=(__uint128_t)(r)[4]-(a)[4]-_b; (r)[4]=(uint64_t)_d; } while(0)

#define NEG5(a) do { \
    __uint128_t _c=1; \
    _c+=~(a)[0]; (a)[0]=(uint64_t)_c; _c>>=64; \
    _c+=~(a)[1]; (a)[1]=(uint64_t)_c; _c>>=64; \
    _c+=~(a)[2]; (a)[2]=(uint64_t)_c; _c>>=64; \
    _c+=~(a)[3]; (a)[3]=(uint64_t)_c; _c>>=64; \
    _c+=~(a)[4]; (a)[4]=(uint64_t)_c; } while(0)

#define SHR62(a) do { \
    (a)[0]=((a)[0]>>62)|((a)[1]<<2); \
    (a)[1]=((a)[1]>>62)|((a)[2]<<2); \
    (a)[2]=((a)[2]>>62)|((a)[3]<<2); \
    (a)[3]=((a)[3]>>62)|((a)[4]<<2); \
    (a)[4]=(int64_t)(a)[4]>>62; } while(0)

#define IS_ZERO(a) (((a)[0]|(a)[1]|(a)[2]|(a)[3]|(a)[4])==0)
#define IS_NEG(a) ((int64_t)(a)[4]<0)

static inline int cmp_ge(const uint64_t* a, const uint64_t* b) {
    for (int i = 4; i >= 0; i--) { if (a[i]!=b[i]) return a[i]>b[i]; }
    return 1;
}

// r = m*a (signed m)
#define SMUL(r,a,m) do { \
    uint64_t _am = (m)>=0 ? (m) : -(m); UMUL(r,a,_am); if((m)<0) NEG5(r); } while(0)

static void fast_modinv(uint64_t* result, const uint64_t* a) {
    uint64_t u[5], v[5], r[5], s[5], t1[5], t2[5], t[5];
    
    u[0]=FI_P[0]; u[1]=FI_P[1]; u[2]=FI_P[2]; u[3]=FI_P[3]; u[4]=0;
    v[0]=a[0]; v[1]=a[1]; v[2]=a[2]; v[3]=a[3]; v[4]=0;
    r[0]=r[1]=r[2]=r[3]=r[4]=0;
    s[0]=1; s[1]=s[2]=s[3]=s[4]=0;
    
    while (!IS_ZERO(v)) {
        int pos = 4;
        while (pos >= 1 && (u[pos]|v[pos]) == 0) pos--;
        
        uint64_t uh, vh;
        if (pos == 0) { uh = u[0]; vh = v[0]; }
        else {
            int sh = __builtin_clzll(u[pos]|v[pos]);
            uh = sh ? (u[pos]<<sh)|(u[pos-1]>>(64-sh)) : u[pos];
            vh = sh ? (v[pos]<<sh)|(v[pos-1]>>(64-sh)) : v[pos];
        }
        
        uint64_t u0 = u[0], v0 = v[0];
        int bc = 62;
        int64_t uu=1, uv=0, vu=0, vv=1;
        
        while (bc > 0) {
            int z = __builtin_ctzll(v0|(1ULL<<bc));
            z = z>bc ? bc : z;
            vh>>=z; v0>>=z; uu<<=z; uv<<=z; bc-=z;
            if (bc <= 0) break;
            if (vh < uh) {
                uint64_t tw=uh; uh=vh; vh=tw; tw=u0; u0=v0; v0=tw;
                int64_t ti=uu; uu=vu; vu=ti; ti=uv; uv=vv; vv=ti;
            }
            vh-=uh; v0-=u0; vu-=uu; vv-=uv;
        }
        
        // new_u = uu*u + uv*v, new_v = vu*u + vv*v
        uint64_t uo[5]={u[0],u[1],u[2],u[3],u[4]};
        uint64_t vo[5]={v[0],v[1],v[2],v[3],v[4]};
        
        SMUL(t1,uo,uu); SMUL(t2,vo,uv); ADD5(t1,t2);
        u[0]=t1[0]; u[1]=t1[1]; u[2]=t1[2]; u[3]=t1[3]; u[4]=t1[4];
        
        SMUL(t1,uo,vu); SMUL(t2,vo,vv); ADD5(t1,t2);
        v[0]=t1[0]; v[1]=t1[1]; v[2]=t1[2]; v[3]=t1[3]; v[4]=t1[4];
        
        if (IS_NEG(u)) { NEG5(u); uu=-uu; uv=-uv; }
        if (IS_NEG(v)) { NEG5(v); vu=-vu; vv=-vv; }
        
        // new_r = uu*r + uv*s, new_s = vu*r + vv*s
        uint64_t ro[5]={r[0],r[1],r[2],r[3],r[4]};
        uint64_t so[5]={s[0],s[1],s[2],s[3],s[4]};
        
        SMUL(t1,ro,uu); SMUL(t2,so,uv); ADD5(t1,t2);
        r[0]=t1[0]; r[1]=t1[1]; r[2]=t1[2]; r[3]=t1[3]; r[4]=t1[4];
        
        SMUL(t1,ro,vu); SMUL(t2,so,vv); ADD5(t1,t2);
        s[0]=t1[0]; s[1]=t1[1]; s[2]=t1[2]; s[3]=t1[3]; s[4]=t1[4];
        
        // Add multiple of P to make r,s divisible by 2^62
        uint64_t r0 = (r[0]*FI_MM64) & FI_MSK62;
        uint64_t s0 = (s[0]*FI_MM64) & FI_MSK62;
        UMUL(t,FI_P,r0); ADD5(r,t);
        UMUL(t,FI_P,s0); ADD5(s,t);
        
        SHR62(u); SHR62(v); SHR62(r); SHR62(s);
    }
    
    while (IS_NEG(r)) ADD5(r,FI_P);
    while (cmp_ge(r,FI_P)) SUB5(r,FI_P);
    
    result[0]=r[0]; result[1]=r[1]; result[2]=r[2]; result[3]=r[3];
}

#endif
