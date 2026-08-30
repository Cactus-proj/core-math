/* Correctly-rounded arc sine function (asinq) in binary128 floating point format.

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
#include <fenv.h> // for FE_INVALID, FE_INEXACT, FE_UNDERFLOW
#include <stdint.h>
#ifdef __x86_64__
#include <x86intrin.h>
#endif
#include <math.h>

// Warning: clang also defines __GNUC__
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#endif

typedef __int128 i128;
typedef unsigned __int128 u128;
typedef uint64_t u64;
typedef int64_t i64;
typedef uint32_t u32;
typedef int32_t i32;
typedef uint64_t u1x64[1];
typedef uint64_t u2x64[2];
typedef uint64_t u3x64[3];
typedef uint64_t u4x64[4];
typedef uint64_t u5x64[5];
typedef uint64_t u6x64[6];
typedef union {
  u128 a;
  i128 as;
  u64 b[2];
  i64 bs[2];
  __float128 f;
} b128u128_u;

// get high part of unsigned 64x64 bit multiplication
static inline u64 mhuu(u64 _a, u64 _b){
  return ((u128)_a*_b)>>64;
}

#if (defined(_WIN32) || defined(__APPLE__))
#define __builtin_addcl __builtin_addcll
#define __builtin_subcl __builtin_subcll
#endif

static inline void mhu3uu3(u3x64 o, u64 y, const u3x64 x){
  u128 xy0 = x[0]*(u128)y;
  u128 xy1 = x[1]*(u128)y;
  u128 xy2 = x[2]*(u128)y;
  u64 c;
  o[0]  = __builtin_addcl(xy1, xy0>>64, 0, &c);
  o[1]  = __builtin_addcl(xy2, xy1>>64, c, &c);
  o[2]  = __builtin_addcl(  0, xy2>>64, c, &c);
}

static inline void mhu3u2u3(u3x64 o, u2x64 y, const u3x64 x){
  u128 x1y0 = x[1]*(u128)y[0];
  u128 x2y0 = x[2]*(u128)y[0];
  u128 x0y1 = x[0]*(u128)y[1];
  u128 x1y1 = x[1]*(u128)y[1];
  u128 x2y1 = x[2]*(u128)y[1];
  x2y0 += x1y0>>64;
  x1y1 += x0y1>>64;
  x2y1 += x1y1>>64;
  u64 c;
  o[0]  = __builtin_addcl(x1y1,     x2y0, 0, &c);
  o[1]  = __builtin_addcl(x2y1, x2y0>>64, c, &c);
  o[2]  = __builtin_addcl(   0, x2y1>>64, c, &c);
}

// get approximate high part of unsigned 128x128 bit multiplication
static inline u128 mhUU(u128 _a, u128 _b){
  b128u128_u a, b, a1b0, a0b1, a1b1;
  a.a = _a;
  b.a = _b;
  a1b0.a = (u128)a.b[1]*b.b[0];
  a0b1.a = (u128)a.b[0]*b.b[1];
  a1b1.a = (u128)a.b[1]*b.b[1];
  a1b1.a += a1b0.b[1];
  a1b1.a += a0b1.b[1];
  return a1b1.a;
}

// o = a + b
static inline void addu1u1u1(u1x64 o, const u1x64 a, const u1x64 b){
  o[0] = a[0]+b[0];
}

static inline void mhu1u1u1(u1x64 o, const u1x64 b, const u1x64 a){
  u128 a1b0 = (u128)a[0]*b[0];
  o[0] = a1b0>>64;
}

// o = a + b
static inline void addu2u2u2(u2x64 o, const u2x64 a, const u2x64 b){
  u64 c;
  o[0] = __builtin_addcl(a[0], b[0], 0, &c);
  o[1] = __builtin_addcl(a[1], b[1], c, &c);
}

static inline void mhu2u2u2(u2x64 o, const u2x64 b, const u2x64 a){
  u64 c0, c1, t, o0, o1;
  u128 a1b0 = (u128)a[1]*b[0];
  o0 = a1b0>>64;

  u128 a0b1 = (u128)a[0]*b[1];
  u128 a1b1 = (u128)a[1]*b[1];
  t  = __builtin_addcl(a1b1, a0b1>>64, 0, &c0);
  o0 = __builtin_addcl(o0, t,  0, &c1);
  o1 = __builtin_addcl(a1b1>>64,   0, c0, &c0);
  o1 = __builtin_addcl(o1, 0, c1, &c1);

  o[0] = o0;
  o[1] = o1;
}

// o = a + b
static inline void addu3u3u3(u3x64 o, const u3x64 a, const u3x64 b){
  u64 c;
  o[0] = __builtin_addcl(a[0], b[0], 0, &c);
  o[1] = __builtin_addcl(a[1], b[1], c, &c);
  o[2] = __builtin_addcl(a[2], b[2], c, &c);
}

// o = a - b
static inline void subu3u3u3(u3x64 o, const u3x64 a, const u3x64 b){
  u64 c;
  o[0] = __builtin_subcl(a[0], b[0], 0, &c);
  o[1] = __builtin_subcl(a[1], b[1], c, &c);
  o[2] = __builtin_subcl(a[2], b[2], c, &c);
}

static inline void mhu3u3u3(u3x64 o, const u3x64 b, const u3x64 a){
  u64 c, o0, o1, o2;
  u128 a1b1 = (u128)a[1]*b[1];
  u128 a2b0 = (u128)a[2]*b[0];
  u128 a0b2 = (u128)a[0]*b[2];
  u128 a2b1 = (u128)a[2]*b[1];
  u128 a1b2 = (u128)a[1]*b[2];
  u128 a2b2 = (u128)a[2]*b[2];

  a2b1 += a2b0>>64;
  a1b2 += a0b2>>64;

  o0 = a1b1>>64;
  o1 = a2b2;
  o2 = a2b2>>64;
  o0 = __builtin_addcl(o0, a2b1, 0, &c);
  o1 = __builtin_addcl(o1, a2b1>>64, c, &c);
  o2 = __builtin_addcl(o2, 0, c, &c);

  o[0] = __builtin_addcl(o0, a1b2, 0, &c);
  o[1] = __builtin_addcl(o1, a1b2>>64, c, &c);
  o[2] = __builtin_addcl(o2,  0, c, &c);
}

// o = a + b
static inline void addu4u4u4(u4x64 o, const u4x64 a, const u4x64 b){
  u64 c;
  o[0] = __builtin_addcl(a[0], b[0], 0, &c);
  o[1] = __builtin_addcl(a[1], b[1], c, &c);
  o[2] = __builtin_addcl(a[2], b[2], c, &c);
  o[3] = __builtin_addcl(a[3], b[3], c, &c);
}

static inline void mhu4u4u4(u4x64 o, const u4x64 b, const u4x64 a){
  u64 c, o0, o1, o2, o3;
  u128 a3b0 = (u128)a[3]*b[0];
  u128 a3b2 = (u128)a[3]*b[2];
  u128 a2b1 = (u128)a[2]*b[1];
  u128 a3b1 = (u128)a[3]*b[1];
  u128 a1b2 = (u128)a[1]*b[2];
  u128 a2b2 = (u128)a[2]*b[2];
  u128 a0b3 = (u128)a[0]*b[3];
  u128 a1b3 = (u128)a[1]*b[3];
  u128 a2b3 = (u128)a[2]*b[3];
  u128 a3b3 = (u128)a[3]*b[3];

  a3b1 += a2b1>>64;
  a2b2 += a1b2>>64;

  o0 = __builtin_addcl(a0b3>>64, a1b3, 0, &c);
  o1 = __builtin_addcl(a1b3>>64, a2b3, c, &c);
  o2 = __builtin_addcl(a2b3>>64, a3b3, c, &c);
  o3 = __builtin_addcl(a3b3>>64,    0, c, &c);

  o0 = __builtin_addcl(o0, a3b0>>64, 0, &c);
  o1 = __builtin_addcl(o1, a3b2, c, &c);
  o2 = __builtin_addcl(o2, a3b2>>64, c, &c);
  o3 = __builtin_addcl(o3, 0, c, &c);

  o0 = __builtin_addcl(o0, a3b1, 0, &c);
  o1 = __builtin_addcl(o1, a3b1>>64, c, &c);
  o2 = __builtin_addcl(o2, 0, c, &c);
  o3 = __builtin_addcl(o3, 0, c, &c);

  o[0] = __builtin_addcl(o0, a2b2, 0, &c);
  o[1] = __builtin_addcl(o1, a2b2>>64, c, &c);
  o[2] = __builtin_addcl(o2, 0, c, &c);
  o[3] = __builtin_addcl(o3, 0, c, &c);
}

// o = a - b
static inline void subu5u5u5(u5x64 o, const u5x64 a, const u5x64 b){
  u64 c;
  o[0] = __builtin_subcl(a[0], b[0], 0, &c);
  o[1] = __builtin_subcl(a[1], b[1], c, &c);
  o[2] = __builtin_subcl(a[2], b[2], c, &c);
  o[3] = __builtin_subcl(a[3], b[3], c, &c);
  o[4] = __builtin_subcl(a[4], b[4], c, &c);
}

// o = a + b
static inline void addu5u5u5(u5x64 o, const u5x64 a, const u5x64 b){
  u64 c;
  o[0] = __builtin_addcl(a[0], b[0], 0, &c);
  o[1] = __builtin_addcl(a[1], b[1], c, &c);
  o[2] = __builtin_addcl(a[2], b[2], c, &c);
  o[3] = __builtin_addcl(a[3], b[3], c, &c);
  o[4] = __builtin_addcl(a[4], b[4], c, &c);
}

static inline void mhu5u5u5(u5x64 o, const u5x64 b, const u5x64 a){
  u64 c0, c1, t, o0, o1, o2, o3, o4;
  u128 a4b0 = (u128)a[4]*b[0];
  o0 = a4b0>>64;

  u128 a3b1 = (u128)a[3]*b[1];
  u128 a4b1 = (u128)a[4]*b[1];
  t  = __builtin_addcl(a4b1, a3b1>>64,  0, &c0);
  o0 = __builtin_addcl(o0, t,  0, &c1);
  o1 = __builtin_addcl(a4b1>>64,    0, c0, &c0);
  o1 = __builtin_addcl(o1, 0, c1, &c1);

  u128 a2b2 = (u128)a[2]*b[2];
  u128 a3b2 = (u128)a[3]*b[2];
  t  = __builtin_addcl(a3b2, a2b2>>64,  0, &c0);
  o0 = __builtin_addcl(o0, t,  0, &c1);
  u128 a4b2 = (u128)a[4]*b[2];
  t  = __builtin_addcl(a4b2, a3b2>>64, c0, &c0);
  o1 = __builtin_addcl(o1, t, c1, &c1);
  o2 = __builtin_addcl(a4b2>>64,    0, c0, &c0);
  o2 = __builtin_addcl(o2, 0, c1, &c1);

  u128 a1b3 = (u128)a[1]*b[3];
  u128 a2b3 = (u128)a[2]*b[3];
  t  = __builtin_addcl(a2b3, a1b3>>64,  0, &c0);
  o0 = __builtin_addcl(o0, t,  0, &c1);
  u128 a3b3 = (u128)a[3]*b[3];
  t  = __builtin_addcl(a3b3, a2b3>>64, c0, &c0);
  o1 = __builtin_addcl(o1, t, c1, &c1);
  u128 a4b3 = (u128)a[4]*b[3];
  t  = __builtin_addcl(a4b3, a3b3>>64, c0, &c0);
  o2 = __builtin_addcl(o2, t, c1, &c1);
  o3 = __builtin_addcl(a4b3>>64,    0, c0, &c0);
  o3 = __builtin_addcl(o3, 0, c1, &c1);

  u128 a0b4 = (u128)a[0]*b[4];
  u128 a1b4 = (u128)a[1]*b[4];
  t  = __builtin_addcl(a1b4, a0b4>>64,  0, &c0);
  o0 = __builtin_addcl(o0, t,  0, &c1);
  u128 a2b4 = (u128)a[2]*b[4];
  t  = __builtin_addcl(a2b4, a1b4>>64, c0, &c0);
  o1 = __builtin_addcl(o1, t, c1, &c1);
  u128 a3b4 = (u128)a[3]*b[4];
  t  = __builtin_addcl(a3b4, a2b4>>64, c0, &c0);
  o2 = __builtin_addcl(o2, t, c1, &c1);
  u128 a4b4 = (u128)a[4]*b[4];
  t  = __builtin_addcl(a4b4, a3b4>>64, c0, &c0);
  o3 = __builtin_addcl(o3, t, c1, &c1);
  o4 = __builtin_addcl(a4b4>>64,    0, c0, &c0);
  o4 = __builtin_addcl(o4, 0, c1, &c1);

  o[0] = o0;
  o[1] = o1;
  o[2] = o2;
  o[3] = o3;
  o[4] = o4;
}

static inline void mhu5u2u5(u5x64 o, const u2x64 b, const u5x64 a){
  u64 c0, c1, t, o0, o1, o2, o3, o4;
  u128 a1b0 = (u128)a[1]*b[0];
  u128 a2b0 = (u128)a[2]*b[0];
  u128 a3b0 = (u128)a[3]*b[0];
  u128 a4b0 = (u128)a[4]*b[0];

  o0  = __builtin_addcl(a2b0, a1b0>>64,  0, &c0);
  o1  = __builtin_addcl(a3b0, a2b0>>64, c0, &c0);
  o2  = __builtin_addcl(a4b0, a3b0>>64, c0, &c0);
  o3  = __builtin_addcl(   0, a4b0>>64, c0, &c0);

  u128 a0b1 = (u128)a[0]*b[1];
  u128 a1b1 = (u128)a[1]*b[1];
  u128 a2b1 = (u128)a[2]*b[1];
  u128 a3b1 = (u128)a[3]*b[1];
  u128 a4b1 = (u128)a[4]*b[1];

  t  = __builtin_addcl(a1b1, a0b1>>64,  0, &c0);
  o0  = __builtin_addcl( o0,        t,  0, &c1);

  t  = __builtin_addcl(a2b1, a1b1>>64, c0, &c0);
  o1  = __builtin_addcl( o1,        t, c1, &c1);

  t  = __builtin_addcl(a3b1, a2b1>>64, c0, &c0);
  o2  = __builtin_addcl( o2,        t, c1, &c1);

  t  = __builtin_addcl(a4b1, a3b1>>64, c0, &c0);
  o3  = __builtin_addcl( o3,        t, c1, &c1);

  t  = __builtin_addcl(   0, a4b1>>64, c0, &c0);
  o4  = __builtin_addcl(  0,        t, c1, &c1);

  o[0] = o0;
  o[1] = o1;
  o[2] = o2;
  o[3] = o3;
  o[4] = o4;
}

static inline void mhu5u1u5(u5x64 o, u64 b0, const u5x64 a){
  u64 c0;
  u128 a0b0 = (u128)a[0]*b0;
  u128 a1b0 = (u128)a[1]*b0;
  u128 a2b0 = (u128)a[2]*b0;
  u128 a3b0 = (u128)a[3]*b0;
  u128 a4b0 = (u128)a[4]*b0;
  
  o[0]  = __builtin_addcl(a1b0, a0b0>>64,  0, &c0);
  o[1]  = __builtin_addcl(a2b0, a1b0>>64, c0, &c0);
  o[2]  = __builtin_addcl(a3b0, a2b0>>64, c0, &c0);
  o[3]  = __builtin_addcl(a4b0, a3b0>>64, c0, &c0);
  o[4]  = __builtin_addcl(   0, a4b0>>64, c0, &c0);
}

static void shrn(int, u64*, int);

static void shln(int n, u64 *a, int k){
  if(__builtin_expect(k<0, 0)){
    shrn(n,a,-k);
  } else {
    int off = k>>6;
    u64 *dst = a + n - 1, *src = dst - off;
    if(__builtin_expect(src >= a, 1)){
      int s = k&63, q = ~k&63;
      u64 c = *src--;
      while(__builtin_expect(src >= a, 1)){
	u64 nt = *src--;
	*dst-- = c<<s|nt>>1>>q;
	c = nt;
      }
      *dst-- = c<<s;
    }
    while(__builtin_expect(dst >= a, 0)) *dst-- = 0;
  }
}

static void shrn(int n, u64 *a, int k){
  if(__builtin_expect(k<0, 0)){
    shln(n,a,-k);
  } else {
    int off = k>>6;
    u64 *src = a + off, *dst = a, *aend = a + n;
    if(__builtin_expect(src < aend, 1)){
      u64 c = *src++;
      int s = k&63;
      if(__builtin_expect(s, 1)){
	int q = -k&63;
	while(__builtin_expect(src < aend, 1)){
	  u64 nt = *src++;
	  *dst++ = c>>s|nt<<q;
	  c = nt;
	}
	*dst++ = c>>s;
      } else {
	while(__builtin_expect(src < aend, 1)){
	  u64 nt = *src++;
	  *dst++ = c;
	  c = nt;
	}
	*dst++ = c;
      }
    }
    while(__builtin_expect(dst < aend, 0)) *dst++ = 0;
  }
}

static inline u128 uq(const u64 *c){
  return (u128)c[1]<<64|c[0];
}

// unsigned 5x64-bit approximate square
static inline void sqrhu5(u5x64 o, const u5x64 a){
  u64 c0, c1, o0, o1, o2, o3, o4, t;
  u128 a4a0 = (u128)a[4]*a[0];
  u128 a3a1 = (u128)a[3]*a[1];
  o0  = __builtin_addcl(a4a0>>64, a3a1>>64,   0, &c0);
  o1  = c0;
  u128 a4a1 = (u128)a[4]*a[1];
  u128 a3a2 = (u128)a[3]*a[2];
  t   = __builtin_addcl(a4a1, a3a2,   0, &c1);
  o0  = __builtin_addcl( o0, t,   c0, &c0);
  t   = __builtin_addcl(a4a1>>64, a3a2>>64,  c1, &c1);
  o1  = __builtin_addcl( o1, t,   c0, &c0);
  u128 a4a2 = (u128)a[4]*a[2];
  o1  = __builtin_addcl( o1, a4a2,   c0, &c0);
  u128 a4a3 = (u128)a[4]*a[3];
  t   = __builtin_addcl(a4a3, a4a2>>64,   c1, &c1);
  o2  = __builtin_addcl( 0, t,   c0, &c0);
  o3  = __builtin_addcl( 0, a4a3>>64,   c1, &c1);

  o0  = __builtin_addcl(o0, o0,  0, &c0);
  o1  = __builtin_addcl(o1, o1, c0, &c0);
  o2  = __builtin_addcl(o2, o2, c0, &c0);
  o3  = __builtin_addcl(o3, o3, c0, &c0);
  o4  = c0;

  u128 a2a2 = (u128)a[2]*a[2];
  u128 a3a3 = (u128)a[3]*a[3];
  u128 a4a4 = (u128)a[4]*a[4];
  o[0]  = __builtin_addcl(o0, a2a2>>64,   0, &c0);
  o[1]  = __builtin_addcl(o1,     a3a3,  c0, &c0);
  o[2]  = __builtin_addcl(o2, a3a3>>64,  c0, &c0);
  o[3]  = __builtin_addcl(o3,     a4a4,  c0, &c0);
  o[4]  = __builtin_addcl(o4, a4a4>>64,  c0, &c0);
}

// unsigned 4x64-bit approximate square
static inline void sqrhu4(u4x64 o, const u4x64 a){
  u64 c0, o0, o1, o2, o3;
  u128 a2a1 = (u128)a[2]*a[1];
  u128 a3a0 = (u128)a[3]*a[0];
  u128 a3a1 = (u128)a[3]*a[1];
  u128 a3a2 = (u128)a[3]*a[2];

  o0  = __builtin_addcl(a3a1, a3a0>>64,  0, &c0);
  o1  = __builtin_addcl(a3a2, a3a1>>64, c0, &c0);
  o2  = __builtin_addcl(   0, a3a2>>64, c0, &c0);

  o0  = __builtin_addcl(o0, a2a1>>64,  0, &c0);
  o1  = __builtin_addcl(o1,        0, c0, &c0);
  o2  = __builtin_addcl(o2,        0, c0, &c0);

  o0  = __builtin_addcl(o0, o0,  0, &c0);
  o1  = __builtin_addcl(o1, o1, c0, &c0);
  o2  = __builtin_addcl(o2, o2, c0, &c0);
  o3  = c0;

  u128 a2a2 = (u128)a[2]*a[2];
  u128 a3a3 = (u128)a[3]*a[3];
  o[0]  = __builtin_addcl(o0,     a2a2,   0, &c0);
  o[1]  = __builtin_addcl(o1, a2a2>>64,  c0, &c0);
  o[2]  = __builtin_addcl(o2,     a3a3,  c0, &c0);
  o[3]  = __builtin_addcl(o3, a3a3>>64,  c0, &c0);
}

// unsigned 2x64-bit approximate square
static inline void sqrhu2(u2x64 o, const u2x64 a){
  u128 a1a0 = (u128)a[1]*a[0];
  u128 a1a1 = (u128)a[1]*a[1];
  a1a1 += a1a0>>63;
  o[0]  = a1a1;
  o[1]  = a1a1>>64;
}

static inline __float128 reinterpret_u128_as_f128(u128 t){
#if defined(__SSE4_1__) && !defined(__clang__)
  // put u128 into xmm register
  __m128i m = {0, 0};
  m = _mm_insert_epi64 (m, t, 0);
  m = _mm_insert_epi64 (m, t>>64, 1);
  __float128 r;
  asm("": "=x"(r): "0"(m));
  return r;
#else
  b128u128_u u = {.a = t};
  return u.f;
#endif
}

static inline u128 reinterpret_f128_as_u128(__float128 z){
#if defined(__SSE4_1__) && !defined(__clang__)
  __m128i t;
  asm("" : "=x" (t) :"0" (z));
  u64 h = _mm_extract_epi64(t, 1);
  u64 l = _mm_extract_epi64(t, 0);
  return ((u128)h<<64)|l;
#else
  b128u128_u u = {.f = z};
  return u.a;
#endif
}

// get full square of unsigned 128 bit argument
static inline void sqrU(u4x64 o, const u2x64 x){
  u128 p10 = (u128)x[1]*x[0];
  u64 c, p10x = p10>>127; p10 <<= 1;
  u128 p00 = (u128)x[0]*x[0];
  u128 p11 = (u128)x[1]*x[1];
  o[0] = p00;
  o[1] = __builtin_addcl(p00>>64, p10, 0, &c);
  o[2] = __builtin_addcl(p10>>64, p11, c, &c);
  o[3] = __builtin_addcl(p11>>64, p10x, c, &c);
}

// get approximate high part of unsigned 128 bit squaring
static inline u128 sqrhU(u128 _a){
  b128u128_u a, a10, a11;
  a.a = _a;
  a10.a = (u128)a.b[1]*a.b[0];
  a10.a >>= 63;
  a11.a = (u128)a.b[1]*a.b[1];
  a11.a += a10.a;
  return a11.a;
}

// get high part of signed 64x64 bit multiplication
static inline i64 mhii(i64 x, i64 y){
  return ((i128)x*y)>>64;
}

// get high 128 bit part of (unsigned 128)x(signed 128) bit
// multiplication with sign mask
static inline i128 mhUIm(u128 _a, i128 _b, u64 mask){
  b128u128_u sub; sub.a = _a;
  sub.b[0] &= mask;
  sub.b[1] &= mask;
  return mhUU(_a,_b) - sub.a;
}

// get high 128 bit part of (unsigned 128)x(signed 128) bit
// multiplication
static inline i128 mhIU(i128 _b, u128 _a){
  return mhUIm(_a,_b,(u64)(_b>>127));
}

// 0 - ordinary number, 1 -- infinity, 2 -- snan, 3 -- qnan
static inline char getclass(u128 x){
  u64 xh = x>>64, xl = x;
  int t = xh>>32 | !!(xh<<32|xl);
  return (t>=0x7fff<<16) + (t>=(0x7fff<<16)+1) + (t>=(0x7fff8<<12));
}

// for low precision approxmation use 64x64->64 bit multiplication
static inline u64 rsqrt9(u64 m){
  static const unsigned c[][4] = {
    {0xffffffff, 0xfffff780, 0xbff55815, 0x9bb5b6e7}, {0xfc0bd889, 0xfa1d6e7d, 0xb8a95a89, 0x938bf8f0},
    {0xf82ec882, 0xf473bea9, 0xb1bf4705, 0x8bed0079}, {0xf467f280, 0xeefff2a1, 0xab309d4a, 0x84cdb431},
    {0xf0b6848c, 0xe9bf46f4, 0xa4f76232, 0x7e24037b}, {0xed19b75e, 0xe4af2628, 0x9f0e1340, 0x77e6ca62},
    {0xe990cdad, 0xdfcd2521, 0x996f9b96, 0x720db8df}, {0xe61b138e, 0xdb16ffde, 0x94174a00, 0x6c913cff},
    {0xe2b7dddf, 0xd68a967b, 0x8f00c812, 0x676a6f92}, {0xdf6689b7, 0xd225ea80, 0x8a281226, 0x62930308},
    {0xdc267bea, 0xcde71c63, 0x8589702c, 0x5e05343e}, {0xd8f7208e, 0xc9cc6948, 0x81216f2e, 0x59bbbcf8},
    {0xd5d7ea91, 0xc5d428ee, 0x7cecdb76, 0x55b1c7d6}, {0xd2c8534e, 0xc1fccbc9, 0x78e8bb45, 0x51e2e592},
    {0xcfc7da32, 0xbe44d94a, 0x75124a0a, 0x4e4b0369}, {0xccd6045f, 0xbaaaee41, 0x7166f40f, 0x4ae66284},
    {0xc9f25c5c, 0xb72dbb69, 0x6de45288, 0x47b19045}, {0xc71c71c7, 0xb3cc040f, 0x6a882804, 0x44a95f5f},
    {0xc453d90f, 0xb0849cd4, 0x67505d2a, 0x41cae1a0}, {0xc1982b2e, 0xad566a85, 0x643afdc8, 0x3f13625c},
    {0xbee9056f, 0xaa406113, 0x6146361f, 0x3c806169}, {0xbc46092e, 0xa7418293, 0x5e70506d, 0x3a0f8e8e},
    {0xb9aedba5, 0xa458de58, 0x5bb7b2b1, 0x37bec572}, {0xb72325b7, 0xa1859022, 0x591adc9a, 0x358c09e2},
    {0xb4a293c2, 0x9ec6bf52, 0x569865a7, 0x33758476}, {0xb22cd56d, 0x9c1b9e36, 0x542efb6a, 0x31797f8a},
    {0xafc19d86, 0x9983695c, 0x51dd5ffb, 0x2f96647a}, {0xad60a1d1, 0x96fd66f7, 0x4fa2687c, 0x2dcab91f},
    {0xab099ae9, 0x9488e64b, 0x4d7cfbc9, 0x2c151d8a}, {0xa8bc441a, 0x92253f20, 0x4b6c1139, 0x2a7449ef},
    {0xa6785b42, 0x8fd1d14a, 0x496eaf82, 0x28e70cc3}, {0xa43da0ae, 0x8d8e042a, 0x4783eba7, 0x276c4900},
    {0xa20bd701, 0x8b594648, 0x45aae80a, 0x2602f493}, {0x9fe2c315, 0x89330ce4, 0x43e2d382, 0x24aa16ec},
    {0x9dc22be4, 0x871ad399, 0x422ae88c, 0x2360c7af}, {0x9ba9da6c, 0x85101c05, 0x40826c88, 0x22262d7b},
    {0x99999999, 0x83126d70, 0x3ee8af07, 0x20f97cd2}, {0x97913630, 0x81215480, 0x3d5d0922, 0x1fd9f714},
    {0x95907eb8, 0x7f3c62ef, 0x3bdedce0, 0x1ec6e994}, {0x93974369, 0x7d632f45, 0x3a6d94a9, 0x1dbfacbb},
    {0x91a55615, 0x7b955498, 0x3908a2be, 0x1cc3a33b}, {0x8fba8a1c, 0x79d2724e, 0x37af80bf, 0x1bd23960},
    {0x8dd6b456, 0x781a2be4, 0x3661af39, 0x1aeae458}, {0x8bf9ab07, 0x766c28ba, 0x351eb539, 0x1a0d21a2},
    {0x8a2345cc, 0x74c813dd, 0x33e61feb, 0x19387676}, {0x88535d90, 0x732d9bdc, 0x32b7823a, 0x186c6f3e},
    {0x8689cc7e, 0x719c7297, 0x3192747d, 0x17a89f21}, {0x84c66df1, 0x70144d19, 0x30769424, 0x16ec9f89},
    {0x83091e6a, 0x6e94e36c, 0x2f63836f, 0x16380fbf}, {0x8151bb87, 0x6d1df079, 0x2e58e925, 0x158a9484},
    {0x7fa023f1, 0x6baf31de, 0x2d567053, 0x14e3d7ba}, {0x7df43758, 0x6a4867d3, 0x2c5bc811, 0x1443880e},
    {0x7c4dd664, 0x68e95508, 0x2b68a346, 0x13a958ab}, {0x7aace2b0, 0x6791be86, 0x2a7cb871, 0x131500ee},
    {0x79113ebc, 0x66416b95, 0x2997c17a, 0x12863c29}, {0x777acde8, 0x64f825a1, 0x28b97b82, 0x11fcc95c},
    {0x75e9746a, 0x63b5b822, 0x27e1a6b4, 0x11786b03}, {0x745d1746, 0x6279f081, 0x2710061d, 0x10f8e6da},
    {0x72d59c46, 0x61449e06, 0x26445f86, 0x107e05ac}, {0x7152e9f4, 0x601591be, 0x257e7b4d, 0x10079327},
    {0x6fd4e793, 0x5eec9e6b, 0x24be2445, 0x0f955da9}, {0x6e5b7d16, 0x5dc9986e, 0x24032795, 0x0f273620},
    {0x6ce6931d, 0x5cac55b7, 0x234d5496, 0x0ebcefdb}, {0x6b7612ec, 0x5b94adb2, 0x229c7cbc, 0x0e56606e},
  };
  // The range [1,2] is splitted into 64 equal sub-ranges and the
  // reciprocal square root is approximated by a cubic polynomial by
  // the minimax method in each subrange. The approximation accuracy
  // fits into 32-33 bits and thus it is natural to round
  // coefficients into 32 bit. The constant coefficient can be
  // rounded to 33 bits since the most significant bit is always 1
  // and implicitly assumed in the table.
  u64 indx = m>>58; // subrange index
  u64 c3 = c[indx][3], c0 = c[indx][0], c1 = c[indx][1], c2 = c[indx][2];
  c0 <<= 31; // to 64 bit with the space for the implicit bit
  c0 |= 1ull<<63; // add implicit bit
  c1 <<= 25; // to 64 bit format
  u64 d = (m<<6)>>32; // local coordinate in the subrange [0, 2^32]
  u64 d2 = ((u64)(d*d))>>32; // square of the local coordinate
  u64 re = c0 + (d2*c2>>13); // even part of the polynomial (positive)
  u64 ro = d*((c1 + ((d2*c3)>>19))>>26)>>6; // odd part of the polynomial (negative)
  u64 r = re - ro; // maximal error < 1.55e-10 and it is less than 2^-32
  // Newton-Raphson first order step to improve accuracy of the result to almost 64 bits
  // r1 = r0 - r0*(r0^2*x - 1)/2
  u64 r2 = mhuu(r,r);
  i64 h = mhuu(m,r2) + r2; // h = r0^2*x - 1
  i64 hr = mhii(h,r>>1); // r0*h/2
  r -= hr;
  if(__builtin_expect(!r, 0)) r--; // adjust in the unlucky case x~1
  return r;
}

// rounded sin(pi/2/72*j) in format prepared for the range reduction
static const short sth[] = {
  11476, 9429, 8096, 7384, 6672, 6053, 5699, 5346, 4995, 4645, 4297,
  4023, 3851, 3680, 3510, 3342, 3174, 3009, 2844, 2681, 2520, 2361,
  2203, 2048, 1894, 1742, 1592, 1445, 1299, 1157, 1016, 878, 742, 609,
  479, 351, 226, 104, -31, -263, -490, -711, -925, -1133, -1335,
  -1531, -1719, -1901, -2105, -2442, -2765, -3074, -3369, -3650,
  -3917, -4240, -4715, -5159, -5574, -5959, -6484, -7134, -7722,
  -8306, -9238, -10046, -11220, -12394, -14139, -16488, -20584
};

// rounded sin(pi/2/72*j)
static const unsigned pth[] = {
  0, 0xb2c0, 0x16560, 0x21800, 0x2ca00, 0x37c00, 0x42d80, 0x4de80,
  0x58f00, 0x63e80, 0x6ed80, 0x79b80, 0x84900, 0x8f500, 0x9a000,
  0xa4a00, 0xaf200, 0xb9a00, 0xc3f00, 0xce400, 0xd8700, 0xe2800,
  0xec700, 0xf6500, 0x100000, 0x109a00, 0x113200, 0x11c800, 0x125b00,
  0x12ed00, 0x137b00, 0x140800, 0x149200, 0x151a00, 0x159f00,
  0x162100, 0x16a100, 0x171e00, 0x179800, 0x180f80, 0x188380,
  0x18f500, 0x196380, 0x19ce80, 0x1a3680, 0x1a9b80, 0x1afd80,
  0x1b5b80, 0x1bb680, 0x1c0e40, 0x1c6280, 0x1cb340, 0x1d0080,
  0x1d4a40, 0x1d9080, 0x1dd340, 0x1e1200, 0x1e4d60, 0x1e84e0,
  0x1eb8c0, 0x1ee8e0, 0x1f1540, 0x1f3de0, 0x1f62a0, 0x1f8390,
  0x1fa0b0, 0x1fb9f0, 0x1fcf50, 0x1fe0d4, 0x1fee76, 0x1ff834,
  0x1ffe0d};

// sqrt(1-pth[j]^2)
static const u5x64 cth[] = {
  {0, 0, 0, 0, 0},
  {0x64d23d92f9ea2771, 0x6d047db3a6e206a7, 0x2953c428c9d7bc56, 0xfb0f1838d5a3a8da, 0xfff06594451d279d},
  {0xe3080aa3bd169d0c, 0x4054eae9456537db, 0x9e5833eae773bc87, 0x6f49ccb466446b89, 0xffc19bc9271a05b3},
  {0x00912c8144374c85, 0x309a44ddae7556ce, 0xc319285201c58b56, 0x3a246e54beb8dac9, 0xff73917b77aa5c47},
  {0x6a7360619e7edb23, 0x0b11b912de529986, 0x5a6995a8cc7d5eac, 0x552a6d7261223cc4, 0xff069a0439cbe651},
  {0x238522ea64680061, 0xf079215ebcd5ca36, 0x290de5267799873e, 0xcab4f035a6bfa060, 0xfe7a55701a8697b0},
  {0x7a92124309b60ef2, 0x4b9dd14818169efc, 0x040ba2edca526f4e, 0xac4772d0d4932b9b, 0xfdcf16b94b0b442e},
  {0x30ff410c35a4b64d, 0xd1d820652a9bc4c6, 0x6d055dc0a4c09600, 0xb4aab1b03d022116, 0xfd04e2530bcbd671},
  {0x03aedba8702f81c3, 0x96da5af0d3ea2f8b, 0xca571fb2cbdee5b9, 0x17c0bd0e66183953, 0xfc1bb125286c39d8},
  {0xfb91abaebc934604, 0xce94d1434f07a0a1, 0x068284240ceeae19, 0x5e9a55955f4b097f, 0xfb143c17bdcc996d},
  {0xab794690d17636d8, 0xc8e308f9bd76c0e2, 0x15fddd7233c65eed, 0xdb301a162cf8115d, 0xf9edc73d6c923555},
  {0x593fea054c7f450e, 0x6b0aa37cffc886a5, 0xd8e480c9ebb134e1, 0x287dce6a9f8ea2ac, 0xf8a922ae090b0799},
  {0x251fef9c8bb46028, 0x493c5f39729df2a1, 0xc745b46aca55d0e8, 0x2062f831fa08f4af, 0xf7454c2c7d12b97a},
  {0xff1ffe460bd9c836, 0xb16e3e5300ab04b0, 0xcc500109643b1f25, 0x1bf03e7ae9dd3a2a, 0xf5c455407155fe59},
  {0x4606e3f3b69b5b9f, 0xaf57d7aca36b7036, 0x23dc96105c0b9d9b, 0x0e73cd55dd020b9f, 0xf4253c1c1a1ae532},
  {0x3e9033c3d73b8ba0, 0x36a1f12dab4107b3, 0xe2bf1fc80065cf4b, 0x067308deced752e6, 0xf267ed62f8b7c795},
  {0xdc78f0c65249d290, 0x3da244d00c619d0d, 0xd5109661f43e9ecd, 0x70a15831a61bca6b, 0xf08f329d05a331b0},
  {0x2e799ef643386021, 0xf50e2c1a3600d3ad, 0x828f074191858249, 0x4f6a116da6359a61, 0xee953e620e55df7b},
  {0xf59cf516cfc8aef3, 0x3994459558477033, 0xcfbcad100d166a9e, 0x6632028b37ff16ad, 0xec832dd8f9584e23},
  {0x8993bed4bc91f71b, 0x2d156655d9d43f98, 0x7386f1c67cf345e5, 0xd03960687c6a9efe, 0xea4f644f48136890},
  {0x6c1be9d31d8598fe, 0x0bbc7599cc98b839, 0x1b0cb81a981ed38d, 0x74401bc33d3f4f33, 0xe800632c0e1d2f2c},
  {0xd77e840a65911d5a, 0x0409ada21048c2b1, 0x54c0567f7377993b, 0xb61bea5c37779619, 0xe59668e019f1e288},
  {0xdf264c217da2460f, 0x4310bc9fc7837821, 0xa5cb92ab27280ab6, 0x03513aa71030fbb2, 0xe311a97671f44f1c},
  {0x8ee20e415370de4e, 0xb85f2ea14495bf6f, 0xa12912f50e813792, 0x0f451e45046e7646, 0xe06dea9a80d1fdf6},
  {0x2485e7ecaf78aedf, 0x639053243722d371, 0x92ec1a6629ed23cc, 0x92ba16b83c5c1dc4, 0xddb3d742c265539d},
  {0x7d45353940ee462b, 0x81cfccf2da1a22d6, 0x4db6039c8f8f6015, 0x83992b6bb6c89121, 0xdada7cc9882f1832},
  {0xb0de57772eac59f6, 0x855009022ef97eb9, 0x8549b723bab4be2d, 0x27e229b54b803697, 0xd7e6403e36d2f1f7},
  {0xe6c36c5ad24b30c1, 0x5258a95fc94e2fb8, 0x63aea18a41e1bf81, 0x2a16383f5c4d5062, 0xd4d7154f370bbe07},
  {0x4f980d926321df45, 0xad92bc366eaf4837, 0x360b41691ba7a5f0, 0x010bd7d48ab25527, 0xd1b27b4ae9008ef2},
  {0xb53f05eae4761ab3, 0x5636614d38c536c1, 0x9cfea8981156d5dd, 0x0bb008a15e850cec, 0xce6d5666b787ca77},
  {0x80638f93f4a2c269, 0x1dcdbcdbbcf14233, 0xae4218b2749a67be, 0x69d2608c3b73e12f, 0xcb190ae896231bce},
  {0xe516cefc6347a5e9, 0x9684ba1ed4646c7a, 0x5f831bdef11f9bbc, 0xc6158124a4ef775f, 0xc7a3b782716d38c8},
  {0x8f005279faa6880d, 0x890f8c878e5f7df9, 0x3485668a4e6e9637, 0xcf01d7d5779f0507, 0xc419953003b5c74b},
  {0x744fe29890f407f9, 0x6f46175fab644dfc, 0xd9330605ec9b4e0a, 0xa322920e8bd7accf, 0xc07416e77d712462},
  {0xab325fd93ce9b6b8, 0x57c460f1a312c3dd, 0x0fcf1122ee32b17c, 0x892c26b0bcfacc8d, 0xbcba10b0d8770ed2},
  {0xcef1e69c4eadb8b3, 0x7f131158279ff4d7, 0x4e6ba5183b584705, 0x8fcd4c453517e8ba, 0xb8ebe30ed9fac186},
  {0x3745f18539bf8772, 0x2cee4ab653475e6e, 0xc28307bea98bb0e7, 0x5cbc46ee3d37b72d, 0xb501e65acbad5ad8},
  {0xf4c3a3a64ddafd51, 0x5e65313f078e13ab, 0xe6ba3e11cdb88afb, 0xf74e3a8ab016619f, 0xb103b40770404b83},
  {0x539c48de71a3cc8e, 0x7f208cb9376741f7, 0xa22a23cb0d6b1468, 0x3fd4750afb3804bb, 0xacf1860da3cfd826},
  {0xbc5eb21dc1786f2d, 0xcaf9195f5c2505ba, 0xf402f48b30c4df2f, 0x4987496d863e1e7c, 0xa8c6fae0bbd09645},
  {0x61f8e4beffcf379f, 0x758e5ddd22563b4f, 0x80d42d484f061220, 0xfbc0967b24f94966, 0xa48d1e8d86992cdc},
  {0xcdf0fca2bcbaff85, 0x201f9ee54d8615fa, 0x7a56fc977f0d0e4c, 0x754bb974869014d6, 0xa03aa9d87a93f00d},
  {0xcae22303aeaf6595, 0xe7b53a3891af1ae5, 0x6de54e5c6a03779d, 0x6cf567b89889fef8, 0x9bd425459273d285},
  {0x9831e7542d85d7ea, 0x611013ca24685535, 0xa3724e71c24123d7, 0xb849ff7aea680179, 0x975eea2b65a9847e},
  {0x101a86b895c7d937, 0xe88c06711a5ab702, 0xfd217ba2d78129ec, 0x20b672429fb3d1dd, 0x92d5d4e3fc1f9cbd},
  {0x4d91c0b5749d4914, 0x8ed80817147623b6, 0xb90bd29f73f58752, 0x5ff44c46ffccf7f3, 0x8e38a46bdb3b3d12},
  {0x2efea8ce593572c2, 0xf37f1d4704fb5c49, 0xf4c2e89353ada913, 0xedbf03f73c1d3162, 0x8986f9b6c7132cff},
  {0x35b06cbf85f84fbb, 0xbf788966cdcd56b0, 0x91e498b6c64670dc, 0xf6e11dfe5f762e4d, 0x84cd81e775b4c098},
  {0x539ba47e0b57d2de, 0x974b441dfa650168, 0xa477f58bc7c8dc47, 0x29dd7114d81c96bf, 0x7fffb96fec8ce447},
  {0x3d5bcfe7068c0ff1, 0x5016bbcd4ef1b8e7, 0x6f1857955007a55c, 0x98ea27d884420256, 0x7b208eaab0b94056},
  {0x846ef313e2e0bf68, 0xabfac2142052bc6c, 0x8496c1983f214586, 0xe132acebbd5846b8, 0x7633823fd4eeef51},
  {0x03681a2fcb57c1e9, 0x999fd17f45cc07f3, 0x411f4be11fd95eb2, 0xa206ce1148e4933c, 0x7138b866dbf2205c},
  {0x701d2b27adfe0fb8, 0x239f4fe8463aee07, 0xe0b2bb9935920222, 0x5d94ac1eeb7461f9, 0x6c3040ff7ae4f3d5},
  {0x064f259d5d960d10, 0xb194cf34ff2bf070, 0x5f46977eac6f5a5c, 0x4367583037f40763, 0x671a12bd2092f0e6},
  {0x739008a332539b2b, 0x34d36b62e94d21ec, 0xf32753bd933f314c, 0x9a57aae41393eb49, 0x61f604a26e818217},
  {0x4cda609f93951dc0, 0xe4f6c5ba31b5c6ed, 0x4fa0c524c392182a, 0x37df71c28f1de586, 0x5cc3c513b7fdb337},
  {0x9a01a8613fd798eb, 0x266e72c3de2e292e, 0x59fec1c23334cf3a, 0x2e578575cc9228df, 0x578dcbb6628e3c6f},
  {0x1f1a07c6d6b6adcd, 0x1eb9dc19b8598724, 0xd9ab93ace24809dc, 0x05dab10687beb878, 0x5246f2e8fd06b80b},
  {0xb8a6a56a834058eb, 0xe60a200e3eb8678a, 0xe50f6f2cd1d3aace, 0xedb16a84844fc0a7, 0x4cfa66f577d9eaf3},
  {0x69aab942d6b77874, 0x9d6df99ad27e5270, 0x80152267d988b733, 0xdbe4096c369e089e, 0x47a24816dd512c95},
  {0x0550f19403d41e0a, 0xbb071ead370af5ff, 0xb9b5812340f07744, 0x64f8889a905e2862, 0x4241a5c4a8543894},
  {0xacd334aee8e9e8a3, 0x92d46ccf664aa8e9, 0x9d79c6111152cb8a, 0x4b40d1af34812df0, 0x3cd8779db76341cd},
  {0x0403e507d395ba75, 0x4c8823c7ba7c9380, 0x0a03c989401a778b, 0x14849e09356ec1dc, 0x37667d4c98bfbccb},
  {0x657bb80e7a20116a, 0xa79b3d94f1414a71, 0x86ab887459d6023c, 0x602a110d5bf3bfc0, 0x31f02728a1820184},
  {0x531dfb48ce58358d, 0xe827b8eac551e9ee, 0x1b75725ed755aa4a, 0xc588e51fb5f4fd50, 0x2c736b07088a07ba},
  {0x90c4aa84e4598669, 0x7dd80f90a7af46ad, 0xa23d2ad47df691ab, 0x4e35d5f602d9b24d, 0x26efff9c8eff17d3},
  {0x06b8ea90e4449667, 0x3b72f4405f272c6f, 0xe689513157c5339a, 0xa73888a873793997, 0x2168e05fb8858700},
  {0xc343b6652a5c735a, 0x18fae9a2c9f7cf27, 0x7e0c3ca1daac2e97, 0xbb8233abbcf0c73a, 0x1bde7b65ebd95dd2},
  {0x3097e3126b116bd9, 0x35782cb810ce1724, 0x5ca4cd631e80444d, 0xd2eff3545c8a56c1, 0x164fbb9bd36b968c},
  {0xe12e48b0cf8a0b8e, 0x402a65318c77466e, 0xfffbc69461af1af2, 0x0d511d6a919beca2, 0x10be2e7affde1a24},
  {0xed8e7549e2530cad, 0x419a20b85f378012, 0x4d48fe6abd0a9a0e, 0x40b937dae10995f7, 0x0b2a9f7bd1a4e3f7},
  {0x0ccf7c9e0c0fe38a, 0xf496f2cb6c0dc6ca, 0x5c67a95d27203004, 0x87e16185128e709b, 0x059591109ebd190a},
};

static const char ind[] = {
  70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70,
  70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 69, 69,
  69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 68,
  68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 67, 67, 67, 67,
  67, 67, 67, 67, 67, 66, 66, 66, 66, 66, 66, 66, 66, 66, 65, 65, 65,
  65, 65, 65, 64, 64, 64, 64, 64, 64, 64, 64, 63, 63, 63, 63, 62, 62,
  62, 62, 62, 61, 61, 61, 61, 61, 60, 60, 60, 60, 59, 59, 59, 58, 58,
  58, 57, 57, 57, 57, 56, 56, 56, 55, 55, 55, 54, 54, 53, 53, 52, 52,
  51, 51, 51, 50, 50, 49, 49, 49, 48, 48, 47, 46, 46, 45, 44, 44, 43,
  42, 42, 41, 41, 40, 39, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 30,
  29, 28, 27, 26, 25, 24, 24, 23, 22, 21, 20, 19, 19, 18, 17, 16, 16,
  15, 14, 13, 13, 12, 11, 11, 10, 10, 10, 9, 9, 9, 8, 8, 7, 7, 7, 6,
  6, 6, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2,
  2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  0, 0, 0, 0, 0, 0
};

// range reduction based on the first 64 bits of a quad precision number 0<x<1
static inline u64 jget(u64 x){
  u64 z = (0x3fffull<<48) - x;
  u64 mz = z<<16;
  long e = z>>48;
  long nz = __builtin_clzll(mz)*(e==0);
  mz <<= nz + (e==0);
  e -= nz;
  long lz = (e<<4|mz>>60) + 161;
  lz *= (lz>=0);
  long j = ind[lz&0xff]*(lz<256);
  long tz = e<<11|mz>>53;
  return j + (tz<sth[j]);
}

// phi0[j] = asin(pth[j]);
static const u6x64 phi0[] = {
  {0, 0, 0, 0, 0},
  {0x31dc03b852fde26e, 0x067073caa9724261, 0x146b2cf786e935f7, 0x13001bf1bb854bc4, 0x2cb0e871e4d3726},
  {0x9fd8e87590496f73, 0xcdd8edd69f613be7, 0xa4cbbf16f87e6782, 0xb7e6ab0fd3c59b0e, 0x595f42cdb812489},
  {0xb3d4031693534f58, 0x5d01a5faaed1d0d3, 0xff3200c1c4f94322, 0x59f6668f9b56acc6, 0x861885ff6c0a35d},
  {0xb4fab3fc67f7a8b9, 0xf5865c9c96d2fa32, 0x29425b02341193a3, 0x48b0673d280eb1c2, 0xb2ba0dd41444021},
  {0x790ddfc5cbe9082f, 0x1c7dc75cedb0a275, 0xc8153f82ea41acef, 0x6b9567f1aad3c76a, 0xdf716a41bbb8b7d},
  {0x6ed0830e8be51bb2, 0xf0e929895ecc82ed, 0x7de84cfcc54fe6f6, 0x76bc77009fd70571, 0x10c23f2fd8e3e9a0},
  {0x2ad1c043c574f910, 0x713ad3d086ff14d2, 0xba270541aa876034, 0x92d9e59dbf55c60d, 0x138d717ba27eeabc},
  {0xb303fd12510f210a, 0x94a653871859e237, 0x0d1502ee17eb1bd7, 0x3739065aec5c06ab, 0x1659054ae2b44a81},
  {0x006d05b30daba4c3, 0x6908be7ddfd7775b, 0xc00a83fef85d99d7, 0x795fa4f368b70f06, 0x192349840a184357},
  {0xa914f322a5010d57, 0x49222451bf9432d3, 0x50da7e2b81928f5c, 0x221d215e79a25603, 0x1bee9de650865623},
  {0x929e0b76f0a42573, 0xda9ba7fc9c3431f1, 0x9174fa977e1baaad, 0xf758a3df6d9553dd, 0x1eb94f058a92e80f},
  {0xcee9d9a2db7de49b, 0xb88aa461c7c4f24c, 0x448f34c5b2254778, 0xcf80327e8e111bfe, 0x2185c42fed9c65a9},
  {0x0cbb8da2a502b8ea, 0xe094e1731ba94e46, 0x97b764bc45ebe177, 0xbe926f0ad70b40fd, 0x245032e12e6a1d6d},
  {0xd3ec0402365050e0, 0xd057a4b66e7da35b, 0x149b8cc2311b2017, 0xec8890b158863303, 0x271b02290466d237},
  {0x1fd4828f5c9b8ca8, 0x319e956b7fde7cc9, 0xc94382ac5f951b31, 0xe06d1bebf0914a3b, 0x29e68bf629a0da11},
  {0x71ae5a9cce6bc46f, 0xf45ac44e54c7356e, 0x4c2591b80b5d94ac, 0x7355bed3fa0a825f, 0x2caeeae98e6df044},
  {0x1614e2e195d53ab6, 0x4f839ef8340df429, 0x040d34ee0e4d94bc, 0xf89ed2ddbcbf4628, 0x2f7cf8d136534aa2},
  {0xf265a4a1fb8335c7, 0xa4778fc22eb6b74e, 0xd853f8dfa8dc313f, 0x1162bc035be61a75, 0x32443631456a4a4b},
  {0x8247e3b00a8a8583, 0x8bdc75bb08da5f10, 0x27e44935f029b5aa, 0xf828f8301f2f6681, 0x3511e7d818e033d1},
  {0x5eaa69eb3673e6b7, 0xded5356713b85cc8, 0xed71cb61a35fb922, 0x6967f0eebe17495f, 0x37ddc249cac0de95},
  {0x56cb01bff8b95b1e, 0x7ae06c0836e66acf, 0xb966728ba24a156d, 0x2431f222087ae262, 0x3aa810220e622ba0},
  {0x0cd54146269826fd, 0xb38d91035609925a, 0x3c90404fa8fdcf10, 0xa93ae7e3b29ab57f, 0x3d711d0d40bc2c8d},
  {0x898dafcf7ea69738, 0x60c82f9ade96a188, 0xd441d9474b741773, 0xf588d3830e2690af, 0x403dc62d61195af3},
  {0xc5c3582884bc019f, 0xab593f8cbe5bde60, 0xb8561a02d8cd4426, 0x96eccb83d59eb445, 0x430548e0b5cd9611},
  {0x536507b58b2d6c0a, 0xbf490c3cfde570cb, 0x4b6bd1e189f08282, 0x624cd14038c8da96, 0x45d126d0c3399240},
  {0x30e1dde73c9aeac0, 0x4d1bffe4ea9a99f2, 0xe5d0081afb2e1436, 0xe295b1b21ac2995c, 0x489d272385822b6e},
  {0x95453966359700ad, 0x17e1392b221445c2, 0x729b7dbd3347f60e, 0xa08f9cf2cc63d953, 0x4b69a4a8a606c031},
  {0xc7743390e65965ff, 0x0b847ab940e4873b, 0x8d1f17662a0b3481, 0xdafccc10fa37a2ea, 0x4e321c56e752580d},
  {0x7b57ce308f292447, 0xd150427a9a863f22, 0xc5e0b9e5677a4e20, 0x37fca23bf3387e48, 0x5100a2ee4edb94b2},
  {0xbda4bd9416c5c7a3, 0x82969bb7db7748d8, 0x00d78b1ec255a12c, 0x32a855887d51994d, 0x53c6bd236475e5cc},
  {0x8c2f292c5227c693, 0x42e7bd886156be32, 0xb0df3bf15143943c, 0xc543c697a208d585, 0x5693b9334fe20fda},
  {0xed129e30b3377786, 0xf685da7f0125cce9, 0x4226edeb7b2b6deb, 0x532a9f496d475cce, 0x595ddcaddd3aedc3},
  {0xc246f045ae5f9921, 0xc469f3cbf90f9ab1, 0xb6ac3d0a4d2b5392, 0x4f4959241985a90a, 0x5c2aaaac17f0e1d6},
  {0xb4cc0a0da35889e0, 0xa4511cd3434d2f34, 0x151fa685f47230d4, 0xfba7987ca6a53862, 0x5ef538076593e829},
  {0x39494cdb29ef9d2f, 0x6798f6ea9d4221d1, 0xd64e85fe6c15a177, 0x2d5b6bbe1108441c, 0x61bdbb8b67421581},
  {0x45aae2f82dee00cc, 0x76e81ce03e84d69a, 0x3316cda9344e35c9, 0xeae47c48f1215e66, 0x648a157110cf21b0},
  {0x41eef34546c95b41, 0x79dfbca51da9087a, 0xb36f5ac6d44d45d1, 0x7c3a89b8d8280209, 0x67551870a5af196b},
  {0x54f1bf06fd9b552d, 0x7e02c07120a7060f, 0x6738fb4ae60936dd, 0x8ddc4b0ddee2054d, 0x6a1f0700be9de55f},
  {0x4e203ffbb54040b7, 0xfe905d1720b87885, 0x986a4823df9a3ca9, 0x61eb119337a3335e, 0x6ceb2fa010c749f6},
  {0xf4a190060d439a24, 0x817d74dcf378cd86, 0x172ea99d30f812a7, 0x6af33f05fd82f72c, 0x6fb3df4b98f240b0},
  {0x9aec3bed5a1c1404, 0x03a65bb59ad7a06a, 0x1846e90fa0de023b, 0x7d2fe76c33e24ecc, 0x727f8d505178bf54},
  {0x6ae899663fdb4f17, 0xaa3655945c870d12, 0x659083b1cd9d70d7, 0x7e7eb442572a626c, 0x754b8b59e6f8c240},
  {0x9af398b34acbe664, 0xe3d6ba0a31b9f6c2, 0x768d232448470674, 0x5141acf6bd77a871, 0x7814d9cc335f7d97},
  {0x7c43fdf5e9ce2813, 0x2de8c2a724b9fa56, 0xae3b7a549e5c78a5, 0x7084ac99e10c434a, 0x7adf11214c0b6503},
  {0xe06fd2c8edc4fd43, 0xb5c5702d7a091fda, 0x8df7819414bb24c6, 0xe968ae72d864e5aa, 0x7daaa2b073c4877f},
  {0xe0558594031f34cb, 0xe3b66c6db1b4677b, 0xa3ac86d4e6c0489a, 0x8833f79ac38b15c1, 0x80780d1e994fb7bb},
  {0x99c89e96e492dbc9, 0xd4129126b1733a9b, 0x725016b221b694e1, 0x75b2b442b022dd71, 0x83402903760fd134},
  {0x9158e49de631a47e, 0xa6c00b85feb1093c, 0xd0872b6328b14761, 0x3dbd94a2b5e37532, 0x860aba7eb47087a9},
  {0xf69af9a28d593e56, 0x1da6495b5f99d1d3, 0x8c9a8af0190b27c8, 0xbfd5796310d923a8, 0x88d65318aae1284d},
  {0x92fad152f7a419fa, 0xa898e37734f35e6f, 0x93294418a7b1eef9, 0xaaa37bf72a9bca36, 0x8ba145b068050a5c},
  {0x9eb0e34d8237d900, 0x248324370f0391a9, 0xf9ab140c3f2319e3, 0xa9a902eb63ec1333, 0x8e6bd976f7ae97c1},
  {0x69046a077a9fb34f, 0xb83240d17b1a5789, 0x97b6ad5de0d13c0f, 0xbac8e70e639b423d, 0x91365e5fef79b4cd},
  {0x6bbdae5c0da83d6c, 0x37712d2db36f9e39, 0xeef92ae1713389c7, 0xb541d1074ddb5287, 0x94012fa0a5db180a},
  {0xbb5676745c77ebbb, 0x57d8473ff0ebece1, 0xaafa85766ff71db6, 0xa7dab323606549d7, 0x96ccb7062a990cae},
  {0x58a33d22eb345c38, 0xa1fbe3b19b08aa77, 0xd8499f6481b60d89, 0x92d1e25a3bc9b0e6, 0x9999717e6e1ec71f},
  {0x261113e4f262af0f, 0x3df0074844f7ac88, 0x4243932d0eabede2, 0xe575abded0b7d4f0, 0x9c621bf8fc82208d},
  {0x406cadf4e2b87105, 0x524e16fdf1d11d7b, 0xaa030b9d01b9fd9b, 0xed08e0cb58f4f54e, 0x9f2e16283b9c5d03},
  {0xd73e6dc68f1113a5, 0x942557aebbe3d9ef, 0xf8f9c3fb6b1b03de, 0x097b5496f5a6e1a0, 0xa1f7c7c61f889ecd},
  {0xc4fa9ec55a050116, 0x63d4f5d564c5b9e6, 0x5820483ea4c7d28f, 0xf2affdc3344f9ced, 0xa4c2a22cf44c08a6},
  {0xc6fe18d269ac4c10, 0xf087df22f5bcc95d, 0xa894adec7b5a152a, 0x78b0e9fb162603ee, 0xa78d60417d3fdd14},
  {0xd65fd0023770eb25, 0x8b6f97871f584da8, 0x1b63f18834d9b341, 0xaf5c7a86d6939813, 0xaa585d25a7e182e7},
  {0xc093aec132b1c4ab, 0xf91f55da8cc955fc, 0xcc5ecaf63081ce7d, 0xdcf0b021b83092b5, 0xad240f36158ad0c8},
  {0xbd41045cdf957eb4, 0x4cf7c3415d74b2e9, 0x01be218cdb35ae43, 0x5dd1c5fec09949c9, 0xafee872faf2e5b19},
  {0x5d33fb200912b7e2, 0x58bc9c09ff4efaa2, 0x331d9bd6ed13a3b4, 0x63858412fbdab72e, 0xb2b928e083506504},
  {0x3b1cdc0b95df2e73, 0x4436f2f0c8996085, 0xc6f98f67ea025fd7, 0xc65a997a7f30b873, 0xb5846f60626b807f},
  {0x403667c077c1fc20, 0x697e62529203f9eb, 0x936eee0e15d0dcea, 0xc22b7c8099952145, 0xb84f2eb5ac3795e7},
  {0x388b200e8525e458, 0xb5ca574fc8a17bb9, 0xbb43e02f8ac0e731, 0x77344f09355e7066, 0xbb19877484ca70a4},
  {0x77d3b12ce16c9c68, 0x822d209b4424efd5, 0x9684a28dad6b791d, 0xce9ca57812ea9dcf, 0xbde45c1866f2d9e6},
  {0x5f01a5224504c357, 0x666c7b378df042a5, 0x3c22aad7fe194a3e, 0x514a5cd169b87306, 0xc0af3b8491d54390},
  {0xcf86b4abffee731f, 0xca3271187934a232, 0x27bd0fddf4ecbdca, 0xe7744018c3407aa7, 0xc37a16c32273901b},
  {0xef9975deaa64c17a, 0x6ba80fa8e07f2cf2, 0x61818a93d068697f, 0xae3e1a514a871a58, 0xc64503961527469c},
};

// approximated normalize 1-x^2
static inline int omx2v2(u4x64 X2, int s, const u2x64 x){
  sqrU(X2, x);
  X2[0] = X2[1]<<1<<(~s&63)|X2[0]>>s;
  X2[0] = ~X2[0];
  X2[1] = X2[2]<<1<<(~s&63)|X2[1]>>s;
  X2[1] = ~X2[1];
  X2[2] = X2[3]<<1<<(~s&63)|X2[2]>>s;
  X2[2] = ~X2[2];
  X2[3] = X2[3]>>s;
  X2[3] = ~X2[3];
  int e = 1;
  if(__builtin_expect(X2[3], 1)){
    int lk = __builtin_clzll(X2[3]);
    X2[3] = X2[3]<<lk|X2[2]>>1>>(~lk&63);
    X2[2] = X2[2]<<lk|X2[1]>>1>>(~lk&63);
    X2[1] = X2[1]<<lk|X2[0]>>1>>(~lk&63);
    X2[0] = X2[0]<<lk;
    e += lk;
  } else {
    int lk = __builtin_clzll(X2[2]);
    X2[3] = X2[2]<<lk|X2[1]>>1>>(~lk&63);
    X2[2] = X2[1]<<lk|X2[0]>>1>>(~lk&63);
    X2[1] = X2[0]<<lk;
    X2[0] = 0;
    e += lk+64;
  }
  return e;
}

static __float128 as_asinq_accurate(__float128);

__float128 cr_asinq(__float128 x) {
  static const u64 c[][2] = {
    {0xaaaaaaaaaaaaaaa9ull, 0xaaaaaaaaaaaaaaaaull},
    {0x333333333333337aull, 0x0013333333333333ull},
    {0xb6db6db6db6dae36ull, 0x000002db6db6db6dull},
    {0x71c71c71c71cfc25ull, 0x000000007c71c71cull},
    {0x2e8ba2e8ba29804eull, 0x000000000016e8baull},
    {0x3b13b13b13ce93b6ull, 0x0000000000000471ull},
    {0xe4cccccccc5f2a5aull, 0x0000000000000000ull},
    {0x002f50f0f1f806dcull, 0x0000000000000000ull},
    {0x000009fef0fec73aull, 0x0000000000000000ull},
    {0x0000000227286573ull, 0x0000000000000000ull},
  };

  unsigned flagp = _mm_getcsr(), oflagp = flagp, rm = flagp&_MM_ROUND_MASK;
  const u64 smsk = 1ull<<63;
  b128u128_u X = {.a = reinterpret_f128_as_u128(x)};
  u64 xsgn = X.b[1]&smsk;
  X.b[1] &= ~smsk; // strip sign
  long xn = X.b[1]>>48;
  // for |x|<0x1.7137449123ef65cdde7f16c56e32p-56 the difference asin(x)-x is less than 0.5 ulp
  if(__builtin_expect(xn<0x3fff-56,0)){ // |x|<0x1p-57
    if(X.a == 0) return x; // x=+/-0 
    if(X.b[1] < 1ull<<48){
      flagp |= FE_UNDERFLOW;
#ifdef CORE_MATH_SUPPORT_ERRNO
      errno = ERANGE; // underflow
#endif
    }
    X.a += (rm != _MM_ROUND_NEAREST)*(!xsgn*(rm==_MM_ROUND_UP) + !!xsgn*(rm==_MM_ROUND_DOWN));
    X.b[1] |= xsgn;
    flagp |= FE_INEXACT;
    // set inexact flag only if it is not set before
    if(__builtin_expect(oflagp!=flagp, 0)) _mm_setcsr(flagp);
    __float128 res = reinterpret_u128_as_f128(X.a); // put into xmm register
    return res;
  }
  if(__builtin_expect(xn>=0x3fff,0)){ // |x|>=1, Inf, NaN
    if(X.b[0]==0&&X.b[1]==(0x3fffull<<48)){
      X.f = 0x1.921fb54442d18469898cc51701b8p+0q;
      X.a += (rm != _MM_ROUND_NEAREST)*(!xsgn*(rm==_MM_ROUND_UP) + !!xsgn*(rm==_MM_ROUND_DOWN));
      X.b[1] |= xsgn;
      flagp |= FE_INEXACT;
      // set inexact flag only if it is not set before
      if(__builtin_expect(oflagp!=flagp, 0)) _mm_setcsr(flagp);
      __float128 res = reinterpret_u128_as_f128(X.a); // put into xmm register
      return res;
    } else {
      char xnan = getclass(X.a);
      if(xnan==2){ // signaling NAN
	flagp |= FE_INVALID;
	if(__builtin_expect(oflagp!=flagp, 0)) _mm_setcsr(flagp);
#if (defined(_WIN32) || defined(__APPLE__))
        return 0.0q / 0.0q;
#else
	return __builtin_nanf128("asinq");
#endif
      } else if(xnan==3) {//quiet NAN
	return x; // propagate nan
      }
#ifdef CORE_MATH_SUPPORT_ERRNO
      errno = EDOM;
#endif
      flagp |= FE_INVALID;
      // set invalid flag only if it is not set before
      if(__builtin_expect(oflagp!=flagp, 0)) _mm_setcsr(flagp);
#if (defined(_WIN32) || defined(__APPLE__))
      return 0.0q / 0.0q;
#else
      return __builtin_nanf128("asinq");
#endif
    }
  }
  u64 j = jget(X.b[1]); // range reduction
  X.b[1] |= 1ull<<48;
  X.a <<= 15;
  u128 t = X.a;
  int nz = 0x3fff-xn;
  u3x64 xc;
  if(j){
    u4x64 X2;
    int e = omx2v2(X2, 2*nz-2, X.b);
    const u64 rsqrt_2[] = {~0ull,0xb504f333f9de6484ull}; // 2^64/sqrt(2)
    u64 rx = X2[3]<<1|X2[2]>>63, r = rsqrt9(rx);
    r = (u128)r*rsqrt_2[e&1]>>64;
    u3x64 SX; mhu3uu3(SX, r, X2+1);
    u3x64 H; mhu3uu3(H, r, SX);
    const int koff = 2, rkoff = 64-koff;
    H[0] = H[0]>>koff|H[1]<<rkoff;
    H[1] = H[1]>>koff|H[2]<<rkoff;
    long hh = H[1];
    u64 h2 = mhii(hh,hh);
    h2 += h2>>1;
    u128 Hh = (u128)H[1]<<64|H[0];
    int lk = (e&1)+koff, rk = (64-lk)&63;
    u128 H2 = (u128)(h2>>rk)<<64|h2<<lk;
    Hh -= H2;
    i128 D = mhIU(Hh, (u128)SX[2]<<64|SX[1]);
    u64 D3s = D>>127;
    u3x64 D3 = {D, D>>64, D3s};
    D3[2] = D3[2]<<lk|D3[1]>>rk;
    D3[1] = D3[1]<<lk|D3[0]>>rk;
    D3[0] = D3[0]<<lk;

    subu3u3u3(SX,SX,D3);
    X.a >>= nz&63;
    mhu3u2u3(xc, X.b, cth[j]+2);
    u64 sj = pth[j];
    int sp = 43-(e>>1);
    if(__builtin_expect(sp>=0, 1)){
      sj <<= sp;
      mhu3uu3(SX, sj, SX);
    } else {
      mhu3uu3(SX, sj, SX);
      rk = -sp&63;
      lk = (64+sp)&63;
      SX[0] = SX[0]>>rk|SX[1]<<lk;
      SX[1] = SX[1]>>rk|SX[2]<<lk;
      SX[2] = SX[2]>>rk;
    }    
    subu3u3u3(xc, xc, SX);
    nz = __builtin_clzll(xc[2]);
    t = (u128)(xc[2]<<nz|xc[1]>>(-nz&63))<<64|(xc[1]<<nz|xc[0]>>(-nz&63));
    addu3u3u3(xc, xc, phi0[j]+2);
  }
  u128 t2 = sqrhU(t);
  u128 t3 = mhUU(t,t2);
  int s2 = 2*(nz-6);
  t2 >>= s2;

  u64 t2h = t2>>64, fl = c[9][0];
  fl = c[8][0] + mhuu(t2h, fl);
  fl = c[7][0] + mhuu(t2h, fl);
  fl = c[6][0] + mhuu(t2h, fl);
  u128 f = (u128)c[5][1]<<64|(c[5][0] + mhuu(t2h, fl));
  int i = 5;
  while(i>0) f = uq(c[--i]) + mhUU(t2, f);
  f = mhUU(t3, f);

  b128u128_u v,dv;
  u64 rnd;
  if(j){
    int sf = 3*nz;
    u3x64 f3;
    f3[2] = f>>64;
    f3[1] = f;
    if(__builtin_expect(sf<64,1)){
      f3[0] = f3[1]<<(-sf&63);
      f3[1] = f3[1]>>sf|f3[2]<<(-sf&63);
      f3[2] = f3[2]>>sf;
    } else if(sf<128){
      sf -= 64;
      f3[0] = f3[1]>>sf|f3[2]<<1<<(~sf&63);
      f3[1] = f3[2]>>sf;
      f3[2] = 0;
    } else if(sf<192){
      sf -= 128;
      f3[0] = f3[2]>>sf;
      f3[2] = f3[1] = 0;
    } else {
      f3[2] = f3[1] = f3[0] = 0;
    }
    addu3u3u3(xc,xc,f3);
    int k = __builtin_clzll(xc[2]);
    rnd = (xc[1]>>(14-k))&1;
    xn = 0x3ffe - k;
    
    u64 Eps = (3*nz-6>57)? 64 : ((1ull<<63)>>(3*nz-6));
    u128 msk = ~(u128)0 >> (k+0x31 + (rm == _MM_ROUND_NEAREST));
    u128 tl = (u128)xc[1]<<64|xc[0];
    tl += Eps;
    tl &= msk;
    if(tl < 2*Eps) return as_asinq_accurate(x);
    v.b[0] = xc[1]>>(15-k)|xc[2]<<(49+k);
    v.b[1] = xc[2]>>(15-k);
  } else {
    int sf = 2*nz+1;
    u4x64 f4;
    f4[3] = f>>64;
    f4[2] = f;
    if(__builtin_expect(sf<64,1)){
      f4[0] = 0;
      f4[1] = f4[2]<<(-sf&63);
      f4[2] = f4[2]>>sf|f4[3]<<(-sf&63);
      f4[3] = f4[3]>>sf;
    } else if(sf<128){
      sf -= 64;
      f4[0] = f4[2]<<(-sf&63);
      f4[1] = f4[2]>>sf|f4[3]<<1<<(~sf&63);
      f4[2] = f4[3]>>sf;
      f4[3] = 0;
    } else if(sf<192){
      sf -= 128;
      f4[0] = f4[2]>>sf|f4[3]<<1<<(~sf&63);
      f4[1] = f4[2]>>sf;
      f4[3] = f4[2] = 0;
    } else {
      f4[3] = f4[2] = f4[1] = f4[0] = 0;
    }
    u64 cr;
    X.a >>= 1;
    f4[2] = __builtin_addcl(X.b[0], f4[2],  0, &cr);
    f4[3] = __builtin_addcl(X.b[1], f4[3], cr, &cr);
    v.b[0] = f4[2];
    v.b[1] = f4[3];
    int k = v.b[1]>>63;
    rnd = (v.b[0]>>(13+k))&1;
    v.a >>= 14+k;
    v.b[1] &= ~0ull>>16;
    xn += k;
    int lk = 64-13-k, rk = 64-lk;
    u64 Th = f4[2]<<lk|f4[1]>>rk, Tl = f4[1]<<lk|f4[0]>>rk;
    u128 Eps = (u128)1<<127; Eps >>= 8+2*nz+k;
    u128 T = (u128)Th<<64|Tl;
    T += Eps;
    if(T<Eps) return as_asinq_accurate(x);
  }
  if(__builtin_expect(rm != _MM_ROUND_NEAREST, 0))
    rnd = !xsgn*(rm==_MM_ROUND_UP) + !!xsgn*(rm==_MM_ROUND_DOWN);
  dv.b[0] = rnd;
  dv.b[1] = (u64)xn<<48|xsgn;
  v.a += dv.a;
  flagp |= FE_INEXACT;
  // set inexact flag only if it is not set before
  if(__builtin_expect(oflagp!=flagp, 0)) _mm_setcsr(flagp);
  __float128 res = reinterpret_u128_as_f128(v.a); // put into xmm register
  return res;
}

// full product o = b0*a
static inline void mu5u1u4(u5x64 o, u64 b0, u4x64 a){
  u128 a0b0 = (u128)a[0]*b0;
  u128 a1b0 = (u128)a[1]*b0;
  u128 a2b0 = (u128)a[2]*b0;
  u128 a3b0 = (u128)a[3]*b0;
  u64 c;
  o[0] = a0b0;
  o[1] = __builtin_addcl(a1b0, a0b0>>64, 0, &c);
  o[2] = __builtin_addcl(a2b0, a1b0>>64, c, &c);
  o[3] = __builtin_addcl(a3b0, a2b0>>64, c, &c);
  o[4] = __builtin_addcl(   0, a3b0>>64, c, &c);
}

// full product o = b0*a
static inline void mu6u1u5(u5x64 o, u64 b0, u4x64 a){
  u128 a0b0 = (u128)a[0]*b0;
  u128 a1b0 = (u128)a[1]*b0;
  u128 a2b0 = (u128)a[2]*b0;
  u128 a3b0 = (u128)a[3]*b0;
  u128 a4b0 = (u128)a[4]*b0;
  u64 c;
  o[0] = a0b0;
  o[1] = __builtin_addcl(a1b0, a0b0>>64, 0, &c);
  o[2] = __builtin_addcl(a2b0, a1b0>>64, c, &c);
  o[3] = __builtin_addcl(a3b0, a2b0>>64, c, &c);
  o[4] = __builtin_addcl(a4b0, a3b0>>64, c, &c);
  o[5] = __builtin_addcl(   0, a4b0>>64, c, &c);
}

// a *= 3;
static inline void mu4x3(u4x64 a){
  u64 c;
  a[0] = __builtin_addcl(a[0], a[1]<<63|a[0]>>1, 0, &c);
  a[1] = __builtin_addcl(a[1], a[2]<<63|a[1]>>1, c, &c);
  a[2] = __builtin_addcl(a[2], a[3]<<63|a[2]>>1, c, &c);
  a[3] = __builtin_addcl(a[3],          a[3]>>1, c, &c);
}

// a *= 5;
static inline void mu3x5(u3x64 a){
  u64 c;
  a[0] = __builtin_addcl(a[0], a[1]<<62|a[0]>>2, 0, &c);
  a[1] = __builtin_addcl(a[1], a[2]<<62|a[1]>>2, c, &c);
  a[2] = __builtin_addcl(a[2],          a[2]>>2, c, &c);
}

// exact normalized 1-x^2
static inline int omx2v3(u4x64 X2, int s, const u2x64 x){
  sqrU(X2, x);
  X2[0] = X2[1]<<1<<(~s&63)|X2[0]>>s;
  X2[1] = X2[2]<<1<<(~s&63)|X2[1]>>s;
  X2[2] = X2[3]<<1<<(~s&63)|X2[2]>>s;
  X2[3] = X2[3]>>s;
  u64 c;
  X2[0] = __builtin_subcl(0, X2[0], 0, &c);
  X2[1] = __builtin_subcl(0, X2[1], c, &c);
  X2[2] = __builtin_subcl(0, X2[2], c, &c);
  X2[3] = __builtin_subcl(0, X2[3], c, &c);
  int e = 1;
  if(__builtin_expect(X2[3], 1)){
    int lk = __builtin_clzll(X2[3]);
    X2[3] = X2[3]<<lk|X2[2]>>1>>(~lk&63);
    X2[2] = X2[2]<<lk|X2[1]>>1>>(~lk&63);
    X2[1] = X2[1]<<lk|X2[0]>>1>>(~lk&63);
    X2[0] = X2[0]<<lk;
    e += lk;
  } else {
    int lk = __builtin_clzll(X2[2]);
    X2[3] = X2[2]<<lk|X2[1]>>1>>(~lk&63);
    X2[2] = X2[1]<<lk|X2[0]>>1>>(~lk&63);
    X2[1] = X2[0]<<lk;
    X2[0] = 0;
    e += lk+64;
  }
  return e;
}

// sqrt(1 - x^2)
static int getcos(u5x64 sq, int ex, u2x64 x){
  u4x64 x2;
  int e = omx2v3(x2, 2*(ex-1), x);
  const u64 rsqrt_2[] = {~0ull,0xb504f333f9de6484ull}; // 2^64/sqrt(2)
  u64 rx = x2[3]<<1|x2[2]>>63, r = rsqrt9(rx);
  r = (u128)r*rsqrt_2[e&1]>>64;
  mu5u1u4(sq, r, x2);
  u6x64 h; mu6u1u5(h, r, sq);
  shrn(6,h,2);
  i64 msk = h[4]; msk>>=63;
  h[4] ^= msk;
  h[3] ^= msk;
  h[2] ^= msk;
  h[1] ^= msk;
  h[0] ^= msk;
  u5x64 h2s; sqrhu4(h2s+1,h+1);
  u2x64 h4s; sqrhu2(h4s,h2s+3);
  u5x64 h3s; mhu3u3u3(h3s+2, h+2, h2s+2);
  mu4x3(h2s+1);
  mu3x5(h3s+2);
  u128 t4u = (u128)h4s[1]<<64|h4s[0];
  t4u += (t4u*3>>5);
  u5x64 t4 = {0,0,0,t4u, t4u>>64};

  h2s[0] = 0;
  shrn(5,h2s,62-(e&1));
  h3s[0] = h3s[1] = 0;
  shrn(5,h3s,59+64-2*(e&1));
  shrn(5,t4,56+128-3*(e&1));

  if(msk){
    addu5u5u5(h,h,h2s);
    addu5u5u5(h,h,h3s);
    addu5u5u5(h,h,t4);
  } else {
    subu5u5u5(h,h,h2s);
    addu5u5u5(h,h,h3s);
    subu5u5u5(h,h,t4);
  }
  
  mu6u1u5(h, r, h);
  u5x64 x2l = {0,x2[0],x2[1],x2[2],x2[3]};
  mhu5u5u5(h,h+1,x2l);  
  shrn(5,h,62-(e&1));
  
  if(!msk){
    subu5u5u5(sq,sq,h);
  } else {
    addu5u5u5(sq,sq,h);
  }
  return e;
}

static inline void evalpoly(u5x64 f, const u5x64 t2){
  static const u64 cp[] = {
    0x1343996b9f42b9f5ull, 0x255e6e351770584dull, 0x00000000000000a3ull, 0x5e2111cba47a2b05ull, 0x000000000005717dull,
    0xa97f20b758a855cdull, 0x000000002ea1bcc9ull, 0x889c99395996e6ceull, 0x00000190cb77f60cull, 0xc7476c854bade5bfull,
    0x000d8137abd89d89ull, 0x97b4ea2813d93845ull, 0x74f4aa383759f229ull, 0x5abb1888e58be523ull, 0x5f1f6db6db6db6dbull,
    0x00000000000003f9ull, 0x9a1160a9ab2539ceull, 0xa8ba2e8ba2e8ba2eull, 0x000000000022bdd3ull, 0x72ec43b868c4b3c0ull,
    0xf7bdef7bdef7bdefull, 0x0000000131683bdeull, 0x4b2852d709bf2295ull, 0x58469ee58469ee58ull, 0x00000a8dd18469eeull,
    0x2d86e53634cafb09ull, 0x684bda12f684bda1ull, 0x005e0b7684bda12full, 0x151d85735049738full, 0xe147ae147ae147aeull,
    0x4d0c7ae147ae147aull, 0x0000000000000003ull, 0x9ba6f1b2735cae39ull, 0x6f4de9bd37a6f4deull, 0xbd37a6f4de9bd37aull,
    0x0000000000001df3ull, 0xcf46c00a8ed8a2e2ull, 0x3cf3cf3cf3cf3cf3ull, 0xf3cf3cf3cf3cf3cfull, 0x000000000112ef3cull,
    0x86baeba7afbb9dd6ull, 0xbca1af286bca1af2ull, 0xa1af286bca1af286ull, 0x00000009fef286bcull, 0xe1e21d9d6b73053dull,
    0xe1e1e1e1e1e1e1e1ull, 0xe1e1e1e1e1e1e1e1ull, 0x00005ea1e1e1e1e1ull, 0x33332cfa4ccaad37ull, 0x3333333333333333ull,
    0x3333333333333333ull, 0x0393333333333333ull, 0x89d89e04e6327ae5ull, 0xd89d89d89d89d89dull, 0x9d89d89d89d89d89ull,
    0x89d89d89d89d89d8ull, 0x0000000000000023ull, 0x8ba2e8b369ee2b14ull, 0xe8ba2e8ba2e8ba2eull, 0x2e8ba2e8ba2e8ba2ull,
    0xa2e8ba2e8ba2e8baull, 0x0000000000016e8bull, 0x8e38e38e78e717bcull, 0x38e38e38e38e38e3ull, 0xe38e38e38e38e38eull,
    0x8e38e38e38e38e38ull, 0x000000000f8e38e3ull, 0x6db6db6db566fac3ull, 0xb6db6db6db6db6dbull, 0xdb6db6db6db6db6dull,
    0x6db6db6db6db6db6ull, 0x000000b6db6db6dbull, 0x99999999999e1925ull, 0x9999999999999999ull, 0x9999999999999999ull,
    0x9999999999999999ull, 0x0009999999999999ull, 0xaaaaaaaaaaaaa521ull, 0xaaaaaaaaaaaaaaaaull, 0xaaaaaaaaaaaaaaaaull,
    0xaaaaaaaaaaaaaaaaull, 0xaaaaaaaaaaaaaaaaull};
  const u64 *ck = cp;
  f[0] = ck[0]; 
  mhu1u1u1(f, f, t2 + 4); addu1u1u1(f, ck += 1, f);
  f[1] = ck[1];
  mhu2u2u2(f, f, t2 + 3); addu2u2u2(f, ck += 2, f);
  mhu2u2u2(f, f, t2 + 3); addu2u2u2(f, ck += 2, f);
  mhu2u2u2(f, f, t2 + 3); addu2u2u2(f, ck += 2, f);
  mhu2u2u2(f, f, t2 + 3); addu2u2u2(f, ck += 2, f);
  mhu2u2u2(f, f, t2 + 3); addu2u2u2(f, ck += 2, f);
  mhu2u2u2(f, f, t2 + 3); addu2u2u2(f, ck += 2, f);
  f[2] = ck[2];
  mhu3u3u3(f, f, t2 + 2); addu3u3u3(f, ck += 3, f);
  mhu3u3u3(f, f, t2 + 2); addu3u3u3(f, ck += 3, f);
  mhu3u3u3(f, f, t2 + 2); addu3u3u3(f, ck += 3, f);
  mhu3u3u3(f, f, t2 + 2); addu3u3u3(f, ck += 3, f);
  mhu3u3u3(f, f, t2 + 2); addu3u3u3(f, ck += 3, f);
  f[3] = ck[3];
  mhu4u4u4(f, f, t2 + 1); addu4u4u4(f, ck += 4, f);
  mhu4u4u4(f, f, t2 + 1); addu4u4u4(f, ck += 4, f);
  mhu4u4u4(f, f, t2 + 1); addu4u4u4(f, ck += 4, f);
  mhu4u4u4(f, f, t2 + 1); addu4u4u4(f, ck += 4, f);
  mhu4u4u4(f, f, t2 + 1); addu4u4u4(f, ck += 4, f);
  mhu4u4u4(f, f, t2 + 1); addu4u4u4(f, ck += 4, f);
  f[4] = ck[4];
  mhu5u5u5(f, f, t2); addu5u5u5(f, ck += 5, f);
  mhu5u5u5(f, f, t2); addu5u5u5(f, ck += 5, f);
  mhu5u5u5(f, f, t2); addu5u5u5(f, ck += 5, f);
  mhu5u5u5(f, f, t2); addu5u5u5(f, ck += 5, f);
  mhu5u5u5(f, f, t2); addu5u5u5(f, ck += 5, f);
}

static inline void cpu5(u5x64 o, const u5x64 a){
  for(int i=0;i<5;i++) o[i] = a[i];
}

__float128 as_asinq_accurate(__float128 x){  
  unsigned flagp = _mm_getcsr(), oflagp = flagp, rm = flagp&_MM_ROUND_MASK;
  const u64 smsk = 1ull<<63;
  b128u128_u X = {.a = reinterpret_f128_as_u128(x)};
  u64 xsgn = X.b[1]&smsk;
  X.b[1] &= ~smsk; // strip sign
  long xn = X.b[1]>>48;
  u64 j = jget(X.b[1]);
  X.b[1] |= 1ull<<48;
  X.a <<= 15;
  u5x64 t = {0,0,0,X.b[0],X.b[1]};
  int nz = 0x3fff-xn;
  u5x64 xc;
  if(j){
    u5x64 sq;
    int e = getcos(sq, nz, X.b);
    X.a >>= nz&63;
    mhu5u2u5(xc, X.b, cth[j]);
    u64 sj = pth[j];
    int sp = 43-(e>>1);
    if(__builtin_expect(sp>=0, 1)) sj <<= sp;
    mhu5u1u5(sq, sj, sq);
    if(__builtin_expect(sp<0, 0)) shrn(5,sq,-sp);
    subu5u5u5(xc, xc, sq);
    cpu5(t, xc);
    for(int i=4;i>=0;i--) if(t[i]) {nz = __builtin_clzll(t[i]) + (4-i)*64; break;}
    shln(5,t,nz);
    addu5u5u5(xc,phi0[j],xc);
  }
  u5x64 t2; sqrhu5(t2,t);
  u5x64 t3; mhu5u5u5(t3,t,t2);
  int s2 = 2*(nz-6)-1;
  shrn(5, t2, s2);
  u5x64 f; evalpoly(f,t2);
  mhu5u5u5(f, t3, f);
  b128u128_u v,dv;
  u64 rnd;
  if(j){
    int sf = 3*nz;
    shrn(5,f,sf);
    addu5u5u5(xc,xc,f);
    int k = __builtin_clzll(xc[4]);
    rnd = (xc[3]>>(14-k))&1;
    xn = 0x3ffe - k;
    v.b[0] = xc[3]>>(15-k)|xc[4]<<(49+k);
    v.b[1] = xc[4]>>(15-k);
  } else {
    int sf = 2*nz+1;
    shrn(5,f,sf);
    X.a >>= 1;
    u64 c;
    f[3] = __builtin_addcl(X.b[0], f[3], 0, &c);
    f[4] = __builtin_addcl(X.b[1], f[4], c, &c);
    v.b[0] = f[3];
    v.b[1] = f[4];
    int k = v.b[1]>>63;
    rnd = (v.b[0]>>(13+k))&1;
    v.a >>= (14+k)&63;
    xn += k;
    v.b[1] &= ~0ull>>16;
  }

  if(__builtin_expect(rm != _MM_ROUND_NEAREST, 0))
    rnd = !xsgn*(rm==_MM_ROUND_UP) + !!xsgn*(rm==_MM_ROUND_DOWN);
  dv.b[0] = rnd;
  dv.b[1] = (u64)xn<<48|xsgn;
  v.a += dv.a;
  flagp |= FE_INEXACT;
  // set inexact flag only if it is not set before
  if(__builtin_expect(oflagp!=flagp, 0)) _mm_setcsr(flagp);
  __float128 res = reinterpret_u128_as_f128(v.a); // put into xmm register
  return res;
}

#ifndef __APPLE__
// somewhat we need to include that for icx and the Intel math library
extern __float128 __asinq (__float128, __float128);

// asinq is called asinf128 in GNU libc, and __asinq in the Intel math library
__float128 asinq(__float128 x) {
#ifdef __INTEL_CLANG_COMPILER
  return __asinq (x);
#else
  return asinf128 (x);
#endif
}
#endif
