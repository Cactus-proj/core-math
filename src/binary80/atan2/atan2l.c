/* Correctly-rounded arc tangent function (atan2l) of two binary80 arguments.

Copyright (c) 2026 Alexei Sibidanov <sibid@uvic.ca>

This file is part of the CORE-MATH project
(https://core-math.gitlabpages.inria.fr/).

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#define _GNU_SOURCE /* to define ...f128 functions */
#include <fenv.h> // for FE_INEXACT, FE_UNDERFLOW
#include <stdint.h>
#ifdef __x86_64__
#  include <x86intrin.h>
#endif
#ifdef CORE_MATH_SUPPORT_ERRNO
#  include <errno.h>
#endif

// Warning: clang also defines __GNUC__
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#endif

typedef __int128 i128;
typedef unsigned __int128 u128;
typedef uint64_t u64;
typedef int64_t i64;
typedef uint32_t u32;
typedef uint64_t u3x64[3];

typedef union {
  long double f;
  struct __attribute__((__packed__))
  {u64 m; u32 e:16; u32 empty:16;};
} b96u96_u;

typedef union {
  u64 b[2];
  u128 a;
} u128_u;

typedef union {
  double f;
  u64 u;
} b64u64_u;

static inline void __attribute__((always_inline)) mhu3u3u2(u3x64 o, const u3x64 a, u128 b){
  u64 b0 = b, b1 = b>>64;
  u128 a1b0 = a[1]*(u128)b0;
  u128 a2b0 = a[2]*(u128)b0;
  u128 a0b1 = a[0]*(u128)b1;
  u128 a1b1 = a[1]*(u128)b1;
  u128 a2b1 = a[2]*(u128)b1;
  u64 c0, c1, t;
  o[0] = __builtin_addcl(a1b1, a0b1>>64,  0, &c0);
  o[1] = __builtin_addcl(a2b1, a1b1>>64, c0, &c0);
  o[2] = __builtin_addcl(   0, a2b1>>64, c0, &c0);
  t    = __builtin_addcl(a2b0, a1b0>>64,  0, &c1);
  o[0] = __builtin_addcl(o[0],        t,  0, &c0);
  t    = __builtin_addcl(   0, a2b0>>64, c1, &c1);
  o[1] = __builtin_addcl(o[1],        t, c0, &c0);
  o[2] = __builtin_addcl(o[2],        0, c0, &c0);
}

// get approximate high part of unsigned 192x192 bit multiplication
static inline void __attribute__((always_inline)) mhu3u3u3(u3x64 o, const u3x64 b, const u3x64 a){
  u128 a2b0 = (u128)a[2]*b[0];
  u128 a2b1 = (u128)a[2]*b[1];
  u128 a2b2 = (u128)a[2]*b[2];

  u64 c0, c1, t, o0, o1, o2;
  o0 = __builtin_addcl(a2b1, a2b0>>64,  0, &c0);
  o1 = __builtin_addcl(a2b2, a2b1>>64, c0, &c0);
  o2 = __builtin_addcl(   0, a2b2>>64, c0, &c0);

  u128 a1b1 = (u128)a[1]*b[1];
  u128 a1b2 = (u128)a[1]*b[2];
  t  = __builtin_addcl(a1b2, a1b1>>64,  0, &c0);
  o0 = __builtin_addcl(  o0,        t,  0, &c1);
  t  = __builtin_addcl(   0, a1b2>>64, c0, &c0);
  o1 = __builtin_addcl(  o1,        t, c1, &c1);
  o2 = __builtin_addcl(  o2,        0, c1, &c1);

  u128 a0b2 = (u128)a[0]*b[2];
  o0 = __builtin_addcl(  o0, a0b2>>64,  0, &c1);
  o1 = __builtin_addcl(  o1,        0, c1, &c1);
  o2 = __builtin_addcl(  o2,        0, c1, &c1);

  o[0] = o0;
  o[1] = o1;
  o[2] = o2;
}

static inline void addu3u3u3(u3x64 o, const u3x64 b, const u3x64 a){
  u64 c, o0, o1, o2;
  o0 = __builtin_addcl(b[0], a[0], 0, &c);
  o1 = __builtin_addcl(b[1], a[1], c, &c);
  o2 = __builtin_addcl(b[2], a[2], c, &c);
  o[0] = o0;
  o[1] = o1;
  o[2] = o2;
}

static inline void subu3u3u3(u3x64 o, const u3x64 b, const u3x64 a){
  u64 c, o0, o1, o2;
  o0 = __builtin_subcl(b[0], a[0], 0, &c);
  o1 = __builtin_subcl(b[1], a[1], c, &c);
  o2 = __builtin_subcl(b[2], a[2], c, &c);
  o[0] = o0;
  o[1] = o1;
  o[2] = o2;
}

static inline void shru3u3(u3x64 o, const u3x64 b, int n){
  if(n<64){
    o[0] = b[0]>>n|(b[1]<<1)<<(~n&63);
    o[1] = b[1]>>n|(b[2]<<1)<<(~n&63);
    o[2] = b[2]>>n;
  } else if(n<128) {
    o[0] = b[1]>>(n&63)|(b[2]<<1)<<(~n&63);
    o[1] = b[2]>>(n&63);
    o[2] = 0;
  } else if(n<192) {
    o[0] = b[2]>>(n&63);
    o[1] = 0;
    o[2] = 0;
  } else {
    o[0] = 0;
    o[1] = 0;
    o[2] = 0;
  }
}

// get high part of unsigned 64x64 bit multiplication
static inline u64 mhuu(u64 _a, u64 _b){
  return ((u128)_a*_b)>>64;
}

// get full product of unsigned 128x128 bit multiplication
static inline u128 mUU(u128 _a, u128 _b, u128 *t){
  u128_u a, b, a1b0, a0b1, a1b1, a0b0;
  a.a = _a;
  b.a = _b;
  a1b0.a = (u128)a.b[1]*b.b[0];
  a0b1.a = (u128)a.b[0]*b.b[1];
  a1b1.a = (u128)a.b[1]*b.b[1];
  a0b0.a = (u128)a.b[0]*b.b[0];
  //     a0b0
  //   a1b0
  //   a0b1
  // a1b1
  unsigned long c;
  a0b0.b[1] = __builtin_addcl(a0b0.b[1], a1b0.b[0], 0, &c);
  a1b1.b[0] = __builtin_addcl(a1b1.b[0], a1b0.b[1], c, &c);
  a1b1.b[1] = __builtin_addcl(a1b1.b[1], 0, c, &c);
  a0b0.b[1] = __builtin_addcl(a0b0.b[1], a0b1.b[0], 0, &c);
  a1b1.b[0] = __builtin_addcl(a1b1.b[0], a0b1.b[1], c, &c);
  a1b1.b[1] = __builtin_addcl(a1b1.b[1], 0, c, &c);
  *t = a0b0.a;
  return a1b1.a;
}

// get approximate high part of unsigned 128x128 bit multiplication
static inline u128 mhUU(u128 _a, u128 _b){
  u128_u a, b, a1b0, a0b1, a1b1;
  a.a = _a;
  b.a = _b;
  a1b0.a = (u128)a.b[1]*b.b[0];
  a0b1.a = (u128)a.b[0]*b.b[1];
  a1b1.a = (u128)a.b[1]*b.b[1];
  a1b1.a += a1b0.b[1];
  a1b1.a += a0b1.b[1];
  return a1b1.a;
}

// get an approximate reciprocal with about 104 bit precision using double precision initial approximation
static inline u128 __attribute__((always_inline)) recipdn(u128 xx0){
  long x = xx0>>64;
  b64u64_u rd = {1.0/x};
  u64 r = (rd.u&(~0ul>>12))|1ul<<52;
  r <<= (rd.u>>52)-0x3c2;
  long dd = (xx0*r)>>64;
  u128 R = r;
  i128 dR = dd*R;
  R <<= 74;
  R -= dR>>39;
  return R;
}

// the accurate path with 192-bit internal precision
long double as_atan2l_accurate(long double y, long double x) {
  unsigned flagp = _mm_getcsr(), oflagp = flagp, rm = flagp&_MM_ROUND_MASK;
  b96u96_u Y = {.f = y}, X = {.f = x};
  int xsgn = X.e>>15, ysgn = Y.e&(1<<15);
  X.e &= 0x7fff; // strip sign
  Y.e &= 0x7fff; // strip sign

  u128 a = X.m|(u128)X.e<<64, b = Y.m|(u128)Y.e<<64;
  i128 dab = a - b, dm = dab>>127;
  dab &= dm;
  a -= dab;
  b += dab;
  int xn = a>>64, yn = b>>64;
  u64 xm = a, ym = b;
  if(__builtin_expect(xn==0, 0)){
    int nz = __builtin_clzll(xm);
    xm <<= nz;
    xn -= nz-1;
  }
  if(__builtin_expect(yn==0, 0)){
    int nz = __builtin_clzll(ym);
    ym <<= nz;
    yn -= nz-1;
  }
  int dn = xn-yn;

  static const unsigned char ind[] = {
    0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 6, 6, 7, 8, 8, 9, 9, 10, 11, 11, 12,
    12, 13, 13, 14, 14, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20,
    21, 21, 22, 22, 22, 23, 23, 24, 24, 25, 25, 25, 26, 26, 27, 27,
    27, 28, 28, 28, 29, 29, 30, 30, 30, 31, 31, 31};
  static const unsigned char rcp[] = {
    255, 252, 248, 244, 240, 237, 234, 230, 227, 224, 221, 218, 215,
    212, 210, 207, 204, 202, 199, 197, 195, 192, 190, 188, 186, 184,
    182, 180, 178, 176, 174, 172, 170, 168, 167, 165, 163, 162, 160,
    159, 157, 156, 154, 153, 151, 150, 148, 147, 146, 144, 143, 142,
    141, 140, 138, 137, 136, 135, 134, 133, 132, 131, 130, 129};
  // a 15 bit approximation of tan(i*pi/4/32)
  static const unsigned short tn[] = {
    0, 0x324, 0x64a, 0x971, 0xc9b, 0xfca, 0x12fd, 0x1636, 0x1976,
    0x1cbe, 0x2010, 0x236c, 0x26d4, 0x2a49, 0x2dcd, 0x3160, 0x3505,
    0x38bd, 0x3c8a, 0x406e, 0x446b, 0x4883, 0x4cb8, 0x510e, 0x5587,
    0x5a26, 0x5eee, 0x63e4, 0x690c, 0x6e6a, 0x7403, 0x79de, 0x8000};

  long k = (ym>>8)*rcp[(xm >> (63-6))&63]>>57;
  if(__builtin_expect(dn<64, 1))
    k >>= dn;
  else
    k = 0;

  long isct = ind[k];
  if(dn<64) isct += (xm >> 15)*tn[isct+1] < (ym>>dn);
  u128 kn, kd;
  if(!isct){
    kn = (u128)ym<<64;
    kd = (u128)xm<<64;
  } else {
    kn = ((u128)ym<<15) - (((u128)xm*tn[isct])<<dn);
    if(__builtin_expect(kn>>127, 0)){
      isct--;
      kn = ((u128)ym<<15) - (((u128)xm*tn[isct])<<dn);
    }
    u64 knh = kn>>64;
    int nzn = (kn)?((knh)?__builtin_clzll(knh):64+__builtin_clzll((u64)kn)):0;
    kn <<= nzn;

    kd = ((u128)ym*tn[isct]) + ((u128)xm<<(dn+15));
    u64 kdh = kd>>64;
    int nzd = __builtin_clzll(kdh);
    kd <<= nzd;
    dn = nzn-nzd;
  }

  u128 kdr = recipdn(kd>>3);
  u128 Hl, Hh = mUU(kd,kdr,&Hl);
  i128 H = Hh<<104|Hl>>24, dr = mhUU(kdr,H) - ((H>>127)&kdr);
  u3x64 dR = {dr>>38,dr>>102,dr>>127};
  u3x64 R = {0, kdr, kdr>>64};
  subu3u3u3(R,R,dR);
  u3x64 T, T2;
  mhu3u3u2(T, R, kn);
  mhu3u3u3(T2, T, T);
  int kz = __builtin_clzll(T2[2]);
  T2[2] = T2[2]<<kz|T2[1]>>(64-kz);
  T2[1] = T2[1]<<kz|T2[0]>>(64-kz);
  T2[0] = T2[0]<<kz;

  int ts = 2*(dn-5) + kz - 4;
  shru3u3(T2,T2,ts);
  static const u3x64 c[] = {
    {0xfffffffffffffffful, 0xfffffffffffffffful, 0xfffffffffffffffful}, // 0
    {0x55555555555554cbul, 0x5555555555555555ul, 0x0015555555555555ul}, // 1
    {0x333333333332f218ul, 0x3333333333333333ul, 0x0000033333333333ul}, // 2
    {0x9249249249186dceul, 0x4924924924924924ul, 0x0000000092492492ul}, // 3
    {0xc71c71c71b423978ul, 0x1c71c71c71c71c71ul, 0x00000000001c71c7ul}, // 4
    {0x45d1745d056d0591ul, 0x745d1745d1745d17ul, 0x00000000000005d1ul}, // 5
    {0xb13b13b08bd10d1bul, 0x3b13b13b13b13b13ul, 0x0000000000000001ul}, // 6
  };
  static const u128_u cl[] = {
    {.b = {0x4444443fa8a4ab96ul, 0x0044444444444444ul}}, // 7
    {.b = {0x0f0f0ef92833a4fful, 0x00000f0f0f0f0f0ful}}, // 8
    {.b = {0x35e50d2e928e031cul, 0x000000035e50d794ul}}, // 9
    {.b = {0xc30c300b96a0a58eul, 0x0000000000c30c30ul}}, // 10
    {.b = {0x90b2150107e26523ul, 0x0000000000002c85ul}}, // 11
    {.b = {0x3d70a24e113d500dul, 0x000000000000000aul}}, // 12
  };
  static const u64 cll[] = {0x025ecf5a3339b5d3ul, 0x00008ca53f70dd91ul};

  u64 fll = mhuu(T2[2], cll[0] - mhuu(T2[2], cll[1]));
  int i = 5;
  u128 T2l = (u128)T2[2]<<64|T2[1], fl = cl[i].a - fll;
  while(--i >= 0) fl = cl[i].a - mhUU(T2l, fl);
  fl = mhUU(T2l, fl);
  u3x64 f = {(u64)fl,(u64)(fl>>64),0};
  i = 6;
  subu3u3u3(f,c[i],f);
  while(--i >=0 ){
    mhu3u3u3(f,T2,f);
    subu3u3u3(f,c[i],f);
  }
  mhu3u3u3(f,T,f);

  if(isct|(dm&1)|xsgn){
    shru3u3(f,f,dn);
    // atan(tn[i])
    static const u3x64 phi0[] = {
      {0, 0, 0},
      {0xb47181e24638fe92, 0x94e685c329b2d773, 0x0191eb5b0f2af2cb},
      {0xf7384a1bcde479ee, 0x3665e6a2207124bc, 0x03245a68934296f4},
      {0x038b8f349e0b9942, 0x60a39999b30013e3, 0x04b650c091b47191},
      {0xb8f721b926dd7969, 0xbc9cc590f04c39b1, 0x0648506548567e1c},
      {0xe5e69a461efee370, 0x98430538072960bc, 0x07dad79d66d396e1},
      {0x98506bfc8617b00e, 0xeef4781de96151b3, 0x096ce71f84443705},
      {0xf1c6b6f3f95d034d, 0xbe30339b35470763, 0x0afef860ced6731a},
      {0x7f78b6932fe4c9aa, 0x573436e5491c3c3b, 0x0c910288a5912f73},
      {0x1576eaf97ffde937, 0x2b63c532aae1baef, 0x0e22f8c260ccbeb2},
      {0xf2cebf4c84ed043f, 0xf9f105fb8919b427, 0x0fb5424b656558f6},
      {0x01f63aedeb6b1188, 0x3e1132674a1a6ece, 0x11474db47224d179},
      {0xeb7eb29db9b819fd, 0xe00180e9012b321a, 0x12d974fae8ccb2fc},
      {0xa94b6c896608a877, 0x0ce9531d42e1282f, 0x146b92651a66881b},
      {0x01533415a5b2003e, 0xf33cb2ee84283a29, 0x15fdebcff69fce1e},
      {0x046e6925e63b9220, 0x1edbbe0a60738519, 0x178fda3cf9a1f86a},
      {0xb36f178c4ec3381d, 0x48b86fb55342a616, 0x192200ca656f9b4e},
      {0xd806ede78bcf8603, 0x7557aecadb39e171, 0x1ab4151b72c17c74},
      {0xc31388ba77526296, 0xadf256ec30dee5f4, 0x1c462f4336d5d87f},
      {0xc1610e738a4c9f6d, 0xc4c9618e94bf1b41, 0x1dd857e3ab6ae652},
      {0x19cdea35fab86a96, 0xa4546bf333ad5f5e, 0x1f6a88026bdc2f47},
      {0xad1715d205e8ce0f, 0x196bdc15af2356b3, 0x20fca91c686855d2},
      {0x1a38ceab9fce7f90, 0xbfdeff67b18287c1, 0x228e9579f321c783},
      {0x7928137ddd0fa064, 0xfd755dd3bdcbab23, 0x2420cf7ee51f46e0},
      {0x49420a6b6791619e, 0x8787fb8a5a45b05b, 0x25b303d3106131a6},
      {0x257de5e0913d9714, 0x13b1717f1dc3721f, 0x2745274cfd47c28c},
      {0x6c0a3ecf5778f283, 0xe8e9142622ebdbb4, 0x28d716f50d9e9584},
      {0x54de998591fc535c, 0xa86e6b5b7a9bae15, 0x2a693870fc67c444},
      {0xb55be89ce49ef929, 0x28b56505dd97d3dd, 0x2bfb77a035071fb6},
      {0xff998a6f2d19532f, 0x9d5322247bb149e5, 0x2d8da0fcd932a561},
      {0xa30583591a2dffbe, 0x1e375bd40e0e3933, 0x2f1faa4f3a17939d},
      {0x6b5f6d06b34c1da6, 0x12b2f428ff7acb76, 0x30b1e91cf0943362},
      {0x4a4093822299f31d, 0x313198a2e0370734, 0x3243f6a8885a308d},
    };
    addu3u3u3(f,phi0[isct],f);
    static const u3x64 off[] = {
      {0,0,0},
      {0x948127044533e63a, 0x62633145c06e0e68, 0x6487ed5110b4611a},
      {0x29024e088a67cc74, 0xc4c6628b80dc1cd1, 0xc90fdaa22168c234},
    };
    int g = dm&1, msk = g^xsgn;
    f[0] ^= -msk;
    f[1] ^= -msk;
    f[2] ^= -msk;
    addu3u3u3(f, off[g+(((!g)&xsgn)*2)], f);
    dn = 0;
  }

  int nz = __builtin_clzll(f[2]);
  f[2] = f[2]<<nz|((f[1]>>1)>>(~nz&63));
  f[1] = f[1]<<nz;

  u64 rnd, r;
  int e = 16384 - nz - dn;
  if(e>0){ // normal numbers
    rnd = f[1]>>63;
    r = f[2];
  } else { // denormal
    if(e>-64){
      rnd = (f[2]>>-e)&1;
      r = (f[2]>>1)>>-e;
    } else { // smaller than denormal so round to zero
      rnd = 0;
      r = 0;
    }
    e  = 0;
  }
  if(__builtin_expect(rm != _MM_ROUND_NEAREST, 0))
    rnd = (rm == _MM_ROUND_UP)*!ysgn + (rm == _MM_ROUND_DOWN)*!!ysgn;
  r += rnd;
  if(__builtin_expect(!(r<<1), 0)){ // here rounding might cross a binade boundary
    if(!e && r == (1ul<<63)){ // denormal to normal rounding
      e = 1;
    }
    if( e && r == 0){ // next binade transition for normal numbers
      r = 1ul<<63;
      ++e;
    }
  }
  if(e == 0){
    feraiseexcept (FE_UNDERFLOW);
#ifdef CORE_MATH_SUPPORT_ERRNO
    errno = ERANGE; // underflow
#endif
  }
  b96u96_u res;
  res.m = r;
  res.e = ysgn|e;

  flagp |= FE_INEXACT;
  if(__builtin_expect(oflagp != flagp, 0)) _mm_setcsr(flagp);

  return res.f;
}

static inline double fasttwosum(double x, double y, double *e){
  double s = x + y, z = s - x;
  *e = y - z;
  return s;
}

static inline double fastsum(double xh, double xl, double yh, double yl, double *e){
  double sl, sh = fasttwosum(xh, yh, &sl);
  *e = (xl + yl) + sl;
  return sh;
}

static inline double muldd(double xh, double xl, double ch, double cl, double *l){
  double ahhh = ch*xh;
  *l = (cl*xh + ch*xl) + __builtin_fma(ch, xh, -ahhh);
  return ahhh;
}

static inline double polydd(double xh, double xl, int n, const double c[][2], double *l){
  int i = n-1;
  double ch = fasttwosum(c[i][0], *l, l), cl = c[i][1] + *l;
  while(--i>=0){
    ch = muldd(xh,xl, ch,cl, &cl);
    ch = fastsum(c[i][0],c[i][1], ch,cl, &cl);
  }
  *l = cl;
  return ch;
}

static long double __attribute__((noinline)) as_atan2l_special(long double y, long double x){
  b96u96_u iy = {.f = y}, ix = {.f = x};
  int xsgn = ix.e>>15, ysgn = iy.e>>15;
  ix.e &= 0x7fff; // strip sign
  iy.e &= 0x7fff; // strip sign
  if(__builtin_expect (iy.e == 0x7fff || ix.e == 0x7fff, 0)){ // NaN or Inf
    if(((iy.m<<1) && iy.e == 0x7fff) || ((ix.m<<1) && ix.e == 0x7fff))
      return y + x; // if y or x is sNaN, returns qNaN and raises invalid
    // Now neither y nor x is NaN, but at least one is +Inf or -Inf
    if(iy.e == 0x7fff && ix.e == 0x7fff){ // both y and x are +/-Inf
      static const long double finf[][2] = {{0x1p-67L, 0x1.921fb54442d1846ap-1L}, {0x1p-64L, 0x1.2d97c7f3321d235p+1L}};
      // atan2 (+/-Inf,-Inf) = +/-3pi/4
      // atan2 (+/-Inf,+Inf) = +/-pi/4
      return __builtin_copysignl(finf[xsgn][1], y) - __builtin_copysignl(finf[xsgn][0], y);
    }
    // now only one of y and x is +/-Inf
    if(ix.e == 0x7fff) {
      if(xsgn)
        return __builtin_copysignl(0x1.921fb54442d1846ap+1L, y) - __builtin_copysignl(0x1p-65L, y);
      // atan2(+/-0,x) = +/-0 for x > 0
      // atan2(+/-y,+Inf) = +/-0 for finite y>0
      return __builtin_copysignl(0.0L, y);
    }
    // now y = +/-Inf
    // atan2(+/-Inf,x) = +/-pi/2 for finite x
    return __builtin_copysignl(0x1.921fb54442d1846ap+0L, y) - __builtin_copysignl(0x1p-66L, y);
  }
  if(__builtin_expect((!iy.e && !iy.m) || (!ix.e && !ix.m), 0)){
    if((!iy.e && !iy.m) && (!ix.e && !ix.m)){
      // atan2(+/-0, +0) = +/-0
      if(!xsgn) return y;
      // atan2(+/-0, -0) = +/-pi
      return (!ysgn) ? 0x1.921fb54442d1846ap+1L - 0x1p-65L : -0x1.921fb54442d1846ap+1L + 0x1p-65L;
    }
    // only one of y and x is zero
    if(!iy.e && !iy.m){
      // atan2(+/-0,x) = +/-0 for x>0
      if(!xsgn) return __builtin_copysignl(0.0L, y);
      // atan2(+/-0,x) = +/-pi for x<0
      return (!ysgn) ? 0x1.921fb54442d1846ap+1L - 0x1p-65L : -0x1.921fb54442d1846ap+1L + 0x1p-65L;
    }
    // now only x is zero
    // atan2(y,+/-0) = -pi/2 for y<0
    // atan2(y,+/-0) = +pi/2 for y>0
    return (!ysgn) ? 0x1.921fb54442d1846ap+0L - 0x1p-66L : -0x1.921fb54442d1846ap+0L + 0x1p-66L;
  }
  return 0;
}

long double cr_atan2l(long double y, long double x){
  b96u96_u iy = {.f = y}, ix = {.f = x};
  int xsgn = ix.e>>15, ysgn = iy.e>>15;
  ix.e &= 0x7fff; // strip sign
  iy.e &= 0x7fff; // strip sign
  if(__builtin_expect (iy.e == 0x7fff || ix.e == 0x7fff, 0)) return as_atan2l_special(y,x); // NaN or Inf
  if(__builtin_expect (!(iy.e|iy.m) || !(ix.e|ix.m), 0)) return as_atan2l_special(y,x); // x=+/-0 or y=+/-0

  u128 a = ix.m|(u128)ix.e<<64, b = iy.m|(u128)iy.e<<64;
  i128 dab = a - b, dm = dab>>127;
  int gt = dm&1;
  dab &= dm;
  a -= dab;
  b += dab;
  int xn = a>>64, yn = b>>64;
  u64 bl = b, al = a;
  if(__builtin_expect(xn==0, 0)){
    int nz = __builtin_clzll(al);
    al <<= nz;
    xn -= nz-1;
  }
  if(__builtin_expect(yn==0, 0)){
    int nz = __builtin_clzll(bl);
    bl <<= nz;
    yn -= nz-1;
  }
  int dn = xn-yn;
  if(__builtin_expect( dn>=53, 0)) return as_atan2l_accurate(y,x);

  b64u64_u yh = {.u = ((bl>>11)&(~0ull>>12))|((0x3ffull-dn)<<52)}, yl = {.u = (bl&0x7ffull)|((0x3f4ull-dn)<<52)}, yloff = {.u = (0x3f4ull-dn)<<52}; yl.f -= yloff.f;
  b64u64_u xh = {.u = ((al>>11)&(~0ull>>12))|(0x3ffull<<52)}, xl = {.u = (al&0x7ffull)|(0x3f4ull<<52)}; xl.f -= 0x1p-11;

  static const double asgn[2] = {1.0, -1.0};
  static const double T2[] = {
    0x0p+0, 0x1p-6, 0x1p-5, 0x1.8p-5, 0x1p-4, 0x1.4p-4, 0x1.8p-4, 0x1.cp-4,
    0x1p-3, 0x1.2p-3, 0x1.4p-3, 0x1.6p-3, 0x1.8p-3, 0x1.ap-3, 0x1.cp-3, 0x1.ep-3,
    0x1p-2, 0x1.1p-2, 0x1.2p-2, 0x1.3p-2, 0x1.4p-2, 0x1.5p-2, 0x1.6p-2, 0x1.7p-2,
    0x1.8p-2, 0x1.9p-2, 0x1.ap-2, 0x1.bp-2, 0x1.cp-2, 0x1.dp-2, 0x1.ep-2, 0x1.fp-2,
    0x1p-1, 0x1.08p-1, 0x1.1p-1, 0x1.18p-1, 0x1.2p-1, 0x1.28p-1, 0x1.3p-1, 0x1.38p-1,
    0x1.4p-1, 0x1.48p-1, 0x1.5p-1, 0x1.58p-1, 0x1.6p-1, 0x1.68p-1, 0x1.7p-1, 0x1.78p-1,
    0x1.8p-1, 0x1.88p-1, 0x1.9p-1, 0x1.98p-1, 0x1.ap-1, 0x1.a8p-1, 0x1.bp-1, 0x1.b8p-1,
    0x1.cp-1, 0x1.c8p-1, 0x1.dp-1, 0x1.d8p-1, 0x1.ep-1, 0x1.e8p-1, 0x1.fp-1, 0x1.f8p-1, 0x1p+0};
  static const double f2[][2] = {
    {0x0p+0, 0x0p+0}, {-0x1.95220c39d4dffp-53, 0x1.fff555bbb73p-7},
    {0x1.2542779d776dep-53, 0x1.ffd55bba976p-6}, {-0x1.6061bbe3de53cp-53, 0x1.7fb818430da4p-5},
    {-0x1.639269b0da47ep-53, 0x1.ff55bb72cfep-5}, {-0x1.4a7663af440f7p-55, 0x1.3f59f0e7c55ap-4},
    {0x1.d1824d59f9e13p-53, 0x1.7ee182602f1p-4}, {0x1.bef71e5340b31p-55, 0x1.be39ebe6f07cp-4},
    {0x1.b8cb225e627dp-53, 0x1.fd5ba9aac2f6p-4}, {0x1.b92de9bac94c2p-53, 0x1.1e1fafb04372p-3},
    {-0x1.d3cb89e62dafdp-54, 0x1.3d6eee8c6627p-3}, {-0x1.882a55960087ap-53, 0x1.5c9811e3ec27p-3},
    {0x1.1347b0b4f881dp-54, 0x1.7b97b4bce5bp-3}, {0x1.873d8079ed0d2p-53, 0x1.9a6a8e96c862p-3},
    {0x1.022f621a5c1cbp-54, 0x1.b90d7529260ap-3}, {0x1.9c648d1534598p-53, 0x1.d77d5df20573p-3},
    {-0x1.4ea9238610a08p-54, 0x1.f5b75f92c80ep-3}, {0x1.2c5c8e721970dp-53, 0x1.09dc597d8636p-2},
    {0x1.30ca4748b1bf9p-57, 0x1.18bf5a30bf178p-2}, {-0x1.20ef9ba6dbf9p-53, 0x1.278372057ef48p-2},
    {0x1.e69c5abb498d2p-53, 0x1.362773707ebc8p-2}, {0x1.a8a86f0ea9311p-54, 0x1.44aa436c2af08p-2},
    {0x1.db5336feef7fp-54, 0x1.530ad9951cd48p-2}, {0x1.9636a3aa3b84p-54, 0x1.614840309cfep-2},
    {0x1.1ce2a8c848b74p-55, 0x1.6f61941e4defp-2}, {-0x1.4b1bbd1ea6db3p-55, 0x1.7d5604b63b3f8p-2},
    {-0x1.4925e8b916e0bp-53, 0x1.8b24d394a1b28p-2}, {0x1.9e6c988fd0a77p-56, 0x1.98cd5454d6b18p-2},
    {-0x1.a49bd836a17p-53, 0x1.a64eec3cc24p-2}, {-0x1.ca3cf09c6b5f8p-53, 0x1.b3a911da65c7p-2},
    {-0x1.cc1ce70934c34p-56, 0x1.c0db4c94ec9fp-2}, {0x1.2e982ddf3872ap-55, 0x1.cde53432c135p-2},
    {-0x1.2ea406ee84d0fp-55, 0x1.dac670561bb5p-2}, {-0x1.de35847c81979p-53, 0x1.e77eb7f175a38p-2},
    {-0x1.a3992dc382a23p-57, 0x1.f40dd0b541418p-2}, {-0x1.b32c949c9d593p-55, 0x1.0039c73c1a40cp-1},
    {-0x1.d5b495f6349e6p-56, 0x1.0657e94db30dp-1}, {-0x1.f34582f6255fep-53, 0x1.0c6145b5b43dcp-1},
    {0x1.ed42511e3f11dp-54, 0x1.1255d9bfbd2a8p-1}, {-0x1.1cef189ff9e7fp-54, 0x1.1835a88be7c14p-1},
    {-0x1.928df287a668fp-58, 0x1.1e00babdefeb4p-1}, {-0x1.e3bde360c7ddbp-53, 0x1.23b71e2cc9e6cp-1},
    {0x1.bd86313ce4fdep-54, 0x1.2958e59308e3p-1}, {-0x1.8e8a85803cc1dp-53, 0x1.2ee628406cbccp-1},
    {-0x1.77ef7641c777fp-54, 0x1.345f01cce37bcp-1}, {0x1.b73ef3389d02fp-53, 0x1.39c391cd41718p-1},
    {0x1.ecf8b492644fp-56, 0x1.3f13fb89e96f4p-1}, {0x1.c1125fd3810c7p-53, 0x1.445065b795b54p-1},
    {0x1.2483350fe548bp-53, 0x1.4978fa3269eep-1}, {0x1.4a33dbeb3796cp-55, 0x1.4e8de5bb6ec04p-1},
    {-0x1.46edd2af69483p-53, 0x1.538f57b89062p-1}, {-0x1.2bcb93b18b52ap-53, 0x1.587d81f732fbcp-1},
    {0x1.0028e4bc5e7cap-57, 0x1.5d58987169b18p-1}, {0x1.ed487acaf1174p-53, 0x1.6220d115d7b8cp-1},
    {-0x1.2dd4dfd7d1777p-53, 0x1.66d663923e088p-1}, {0x1.2bfe3cf3b9d79p-54, 0x1.6b798920b3d98p-1},
    {-0x1.8c34d25aadef6p-56, 0x1.700a7c5784634p-1}, {-0x1.f426acf4d3bdbp-54, 0x1.748978fba8e1p-1},
    {-0x1.afe57dd9ff23p-53, 0x1.78f6bbd5d316p-1}, {-0x1.54fbef0e862abp-54, 0x1.7d528289fa094p-1},
    {0x1.90227758b11bap-54, 0x1.819d0b7158a4cp-1}, {0x1.16b66e7fc8b8cp-53, 0x1.85d69576cc2c4p-1},
    {-0x1.55b9a5e177a1bp-55, 0x1.89ff5ff57f1f8p-1}, {0x1.c27cfaa9f7a14p-53, 0x1.8e17aa99cc05cp-1},
    {0x1.1a62633145c07p-55, 0x1.921fb54442d18p-1}};
  static const double O[8][2] = {
    {0,0}, {0x1.921fb54442d18p+0,0x1.1a62633145c07p-54},
    {0,0}, {-0x1.921fb54442d18p+0,-0x1.1a62633145c07p-54},
    {0x1.921fb54442d18p+1,0x1.1a62633145c07p-53}, {0x1.921fb54442d18p+0,0x1.1a62633145c07p-54},
    {-0x1.921fb54442d18p+1,-0x1.1a62633145c07p-53}, {-0x1.921fb54442d18p+0,-0x1.1a62633145c07p-54}};

  double sgn = asgn[gt^xsgn^ysgn];
  u64 kw = xsgn<<2|ysgn<<1|gt;
  b64u64_u jj = {.f = yh.f/xh.f + (2 + 1/128.)};
  i64 jt = ((jj.u>>(52-7))&127);
  double f0h = f2[jt][1]*sgn + O[kw][0];
  double f0l = f2[jt][0]*sgn + O[kw][1];
  double t0 = T2[jt];
  double dh = yh.f*t0, dl = __builtin_fma(yh.f,t0,-dh), e, rdh;
  dh = fasttwosum(xh.f, dh, &e);
  dl += e + yl.f*t0 + xl.f;
  dh = fasttwosum(dh,dl,&dl);
  rdh = 1/dh;
  double nh = xh.f*t0, nl = __builtin_fma(xh.f,t0,-nh);
  nl -= yl.f - xl.f*t0;
  double dt = yh.f-nh, y1 = dt+nh;
  if(__builtin_expect(y1 == yh.f, 1)){
    nh = fasttwosum(dt, -nl, &nl);
  } else {
    nh = fasttwosum(dt, (yh.f - y1) - nl, &nl);
  }
  double zh = nh * rdh;
  double zl = rdh * (__builtin_fma(dh, -zh, nh) + (nl - (nh*rdh)*dl));
  double z2 = zh*zh, z2l = __builtin_fma(zh,zh,-z2) + 2*zh*zl;

  static const double c[][2] = {{0x1p+0, -0x1.ffeb445336f3cp-81}, {-0x1.5555555555555p-2, -0x1.5055876e9de02p-56}};
  static const double cl[] = {0x1.999999999961ap-3, -0x1.24924904934fp-3, 0x1.c70c7228655edp-4};
  double fl = z2*(cl[0] + z2*(cl[1] + z2*(cl[2]))), fh = polydd(z2,z2l,2,c, &fl);
  zh *= sgn;
  zl *= sgn;
  fh = muldd(zh,zl, fh,fl, &fl);
  fh = fastsum(f0h,f0l,fh,fl,&fl);

  // fails with eps=0.4e-26 and
  // x=-0xb.ac961beb22287cp-10 y=0x8.e1b862d0b5ca25p-7 (rndz)
  const double eps = 1.2e-26, eps0 = 0x1p-80;
  long double res = fh, flu = fl + eps, fll = fl - eps;
  long double ub = res + flu, lb = res + fll;
  if(ub != lb){
    if(jt == 0 && gt == 0 && xsgn == 0 && __builtin_fabs(fh)<0x1.15p-9){
      ub = res + fl;
      lb = res + (fl + eps0*fh);
      if(ub != lb){
	return as_atan2l_accurate(y,x);
      }
      return ub;
    }
    return as_atan2l_accurate(y,x);
  }
  return ub;
}
