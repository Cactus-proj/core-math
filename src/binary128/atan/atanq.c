/* Correctly-rounded arc tangent function (atanq) in binary128 floating point format.

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

static inline u128 mhuU(u64 y, u128 x){
  b128u128_u ux; ux.a = x;
  u128 xy0 = ux.b[0]*(u128)y;
  u128 xy1 = ux.b[1]*(u128)y;
  return xy1 + (xy0>>64);
}

// get approximate high part of unsigned 128 bit squaring
static inline u128 sqrhU(u128 _a){
  b128u128_u a, a10, a11;
  a.a = _a;
  a10.a = (u128)a.b[1]*a.b[0];
  a11.a = (u128)a.b[1]*a.b[1];
  a11.a += a10.b[1];
  a11.a += a10.b[1];
  return a11.a;
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

#if (defined(_WIN32) || defined(__APPLE__))
#define __builtin_addcl __builtin_addcll
#define __builtin_subcl __builtin_subcll
#endif

// o = a - b
static inline void subu2u2u2(u2x64 o, const u2x64 a, const u2x64 b){
  u64 c;
  o[0] = __builtin_subcl(a[0], b[0], 0, &c);
  o[1] = __builtin_subcl(a[1], b[1], c, &c);
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

// o = a - b
static inline void subu4u4u4(u4x64 o, const u4x64 a, const u4x64 b){
  u64 c;
  o[0] = __builtin_subcl(a[0], b[0], 0, &c);
  o[1] = __builtin_subcl(a[1], b[1], c, &c);
  o[2] = __builtin_subcl(a[2], b[2], c, &c);
  o[3] = __builtin_subcl(a[3], b[3], c, &c);
}

static inline void mhu4u4u4(u4x64 o, const u4x64 b, const u4x64 a){
  //static void __attribute__((noinline)) mhu4u4u4(u4x64 o, const u4x64 b, const u4x64 a){
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

// o = a + b
static inline void addu6u6u6(u6x64 o, const u6x64 a, const u6x64 b){
  u64 c;
  o[0] = __builtin_addcl(a[0], b[0], 0, &c);
  o[1] = __builtin_addcl(a[1], b[1], c, &c);
  o[2] = __builtin_addcl(a[2], b[2], c, &c);
  o[3] = __builtin_addcl(a[3], b[3], c, &c);
  o[4] = __builtin_addcl(a[4], b[4], c, &c);
  o[5] = __builtin_addcl(a[5], b[5], c, &c);
}

// o = a - b
static inline void subu6u6u6(u6x64 o, const u6x64 a, const u6x64 b){
  u64 c;
  o[0] = __builtin_subcl(a[0], b[0], 0, &c);
  o[1] = __builtin_subcl(a[1], b[1], c, &c);
  o[2] = __builtin_subcl(a[2], b[2], c, &c);
  o[3] = __builtin_subcl(a[3], b[3], c, &c);
  o[4] = __builtin_subcl(a[4], b[4], c, &c);
  o[5] = __builtin_subcl(a[5], b[5], c, &c);
}

//static void __attribute__((noinline)) mhu6u6u6(u6x64 o, const u6x64 b, const u6x64 a){
static inline void mhu6u6u6(u6x64 o, const u6x64 b, const u6x64 a){
  u64 c0, c1, t, o0, o1, o2, o3, o4, o5;
  u128 a5b0 = (u128)a[5]*b[0];
  o0 = a5b0>>64;

  u128 a4b1 = (u128)a[4]*b[1];
  u128 a5b1 = (u128)a[5]*b[1];
  t  = __builtin_addcl(a5b1, a4b1>>64,  0, &c0);
  o0 = __builtin_addcl(o0, t,  0, &c1);
  o1 = __builtin_addcl(a5b1>>64,    0, c0, &c0);
  o1 = __builtin_addcl(o1, 0, c1, &c1);

  u128 a3b2 = (u128)a[3]*b[2];
  u128 a4b2 = (u128)a[4]*b[2];
  t  = __builtin_addcl(a4b2, a3b2>>64,  0, &c0);
  o0 = __builtin_addcl(o0, t,  0, &c1);
  u128 a5b2 = (u128)a[5]*b[2];
  t  = __builtin_addcl(a5b2, a4b2>>64, c0, &c0);
  o1 = __builtin_addcl(o1, t, c1, &c1);
  o2 = __builtin_addcl(a5b2>>64,    0, c0, &c0);
  o2 = __builtin_addcl(o2, 0, c1, &c1);

  u128 a1b3 = (u128)a[2]*b[3];
  u128 a2b3 = (u128)a[3]*b[3];
  t  = __builtin_addcl(a2b3, a1b3>>64,  0, &c0);
  o0 = __builtin_addcl(o0, t,  0, &c1);
  u128 a3b3 = (u128)a[4]*b[3];
  t  = __builtin_addcl(a3b3, a2b3>>64, c0, &c0);
  o1 = __builtin_addcl(o1, t, c1, &c1);
  u128 a4b3 = (u128)a[5]*b[3];
  t  = __builtin_addcl(a4b3, a3b3>>64, c0, &c0);
  o2 = __builtin_addcl(o2, t, c1, &c1);
  o3 = __builtin_addcl(a4b3>>64,    0, c0, &c0);
  o3 = __builtin_addcl(o3, 0, c1, &c1);

  u128 a1b4 = (u128)a[1]*b[4];
  u128 a2b4 = (u128)a[2]*b[4];
  t  = __builtin_addcl(a2b4, a1b4>>64,  0, &c0);
  o0 = __builtin_addcl(o0, t,  0, &c1);
  u128 a3b4 = (u128)a[3]*b[4];
  t  = __builtin_addcl(a3b4, a2b4>>64, c0, &c0);
  o1 = __builtin_addcl(o1, t, c1, &c1);
  u128 a4b4 = (u128)a[4]*b[4];
  t  = __builtin_addcl(a4b4, a3b4>>64, c0, &c0);
  o2 = __builtin_addcl(o2, t, c1, &c1);
  u128 a5b4 = (u128)a[5]*b[4];
  t  = __builtin_addcl(a5b4, a4b4>>64, c0, &c0);
  o3 = __builtin_addcl(o3, t, c1, &c1);
  o4 = __builtin_addcl(a5b4>>64,    0, c0, &c0);
  o4 = __builtin_addcl(o4, 0, c1, &c1);

  u128 a0b5 = (u128)a[0]*b[5];
  u128 a1b5 = (u128)a[1]*b[5];
  t  = __builtin_addcl(a1b5, a0b5>>64,  0, &c0);
  o0 = __builtin_addcl(o0, t,  0, &c1);
  u128 a2b5 = (u128)a[2]*b[5];
  t  = __builtin_addcl(a2b5, a1b5>>64, c0, &c0);
  o1 = __builtin_addcl(o1, t, c1, &c1);
  u128 a3b5 = (u128)a[3]*b[5];
  t  = __builtin_addcl(a3b5, a2b5>>64, c0, &c0);
  o2 = __builtin_addcl(o2, t, c1, &c1);
  u128 a4b5 = (u128)a[4]*b[5];
  t  = __builtin_addcl(a4b5, a3b5>>64, c0, &c0);
  o3 = __builtin_addcl(o3, t, c1, &c1);
  u128 a5b5 = (u128)a[5]*b[5];
  t  = __builtin_addcl(a5b5, a4b5>>64, c0, &c0);
  o4 = __builtin_addcl(o4, t, c1, &c1);
  o5 = __builtin_addcl(a5b5>>64,    0, c0, &c0);
  o5 = __builtin_addcl(o5, 0, c1, &c1);

  o[0] = o0;
  o[1] = o1;
  o[2] = o2;
  o[3] = o3;
  o[4] = o4;
  o[5] = o5;
}

static void mu5u2u3(u64 *o, const u64 *b, const u64 *a){
  u128 T; u64 t0,t1,tt,o0,o1,o2,o3,o4,c,c0,c1;
  T = (u128)b[0]*a[0]; t1 = T>>64; t0 = T;
  o0 = t0;
  tt = t1;
  T = (u128)b[0]*a[1]; t1 = T>>64; t0 = T;
  o1 = __builtin_addcl(tt, t0, 0, &c);
  tt = t1;
  T = (u128)b[0]*a[2]; t1 = T>>64; t0 = T;
  o2 = __builtin_addcl(tt, t0, c, &c);
  o3 = __builtin_addcl(0, t1, c, &c);

  T = (u128)b[1]*a[0]; t1 = T>>64; t0 = T;
  o1 = __builtin_addcl(o1, t0, 0, &c0);
  tt = t1;
  T = (u128)b[1]*a[1]; t1 = T>>64; t0 = T;
  t0 = __builtin_addcl(tt, t0, 0, &c1);
  o2 = __builtin_addcl(o2, t0, c0, &c0);
  tt = t1;
  T = (u128)b[1]*a[2]; t1 = T>>64; t0 = T;
  t0 = __builtin_addcl(tt, t0, c1, &c1);
  o3 = __builtin_addcl(o3, t0, c0, &c0);
  t1 = __builtin_addcl(0, t1, c1, &c1);
  o4 = __builtin_addcl(0, t1, c0, &c0);

  o[0] = o0;
  o[1] = o1;
  o[2] = o2;
  o[3] = o3;
  o[4] = o4;
}

static void mu7u5u2(u64 *o, const u64 *b, const u64 *a){
  u128 T; u64 t0,t1,tt,o0,o1,o2,o3,o4,o5,o6,c,c0,c1;
  T = (u128)b[0]*a[0]; t1 = T>>64; t0 = T;
  o0 = t0;
  tt = t1;
  T = (u128)b[0]*a[1]; t1 = T>>64; t0 = T;
  o1 = __builtin_addcl(tt, t0, 0, &c);
  o2 = __builtin_addcl(0, t1, c, &c);

  T = (u128)b[1]*a[0]; t1 = T>>64; t0 = T;
  o1 = __builtin_addcl(o1, t0, 0, &c0);
  tt = t1;
  T = (u128)b[1]*a[1]; t1 = T>>64; t0 = T;
  t0 = __builtin_addcl(tt, t0, 0, &c1);
  o2 = __builtin_addcl(o2, t0, c0, &c0);
  t1 = __builtin_addcl(0, t1, c1, &c1);
  o3 = __builtin_addcl(0, t1, c0, &c0);

  T = (u128)b[2]*a[0]; t1 = T>>64; t0 = T;
  o2 = __builtin_addcl(o2, t0, 0, &c0);
  tt = t1;
  T = (u128)b[2]*a[1]; t1 = T>>64; t0 = T;
  t0 = __builtin_addcl(tt, t0, 0, &c1);
  o3 = __builtin_addcl(o3, t0, c0, &c0);
  t1 = __builtin_addcl(0, t1, c1, &c1);
  o4 = __builtin_addcl(0, t1, c0, &c0);

  T = (u128)b[3]*a[0]; t1 = T>>64; t0 = T;
  o3 = __builtin_addcl(o3, t0, 0, &c0);
  tt = t1;
  T = (u128)b[3]*a[1]; t1 = T>>64; t0 = T;
  t0 = __builtin_addcl(tt, t0, 0, &c1);
  o4 = __builtin_addcl(o4, t0, c0, &c0);
  t1 = __builtin_addcl(0, t1, c1, &c1);
  o5 = __builtin_addcl(0, t1, c0, &c0);

  T = (u128)b[4]*a[0]; t1 = T>>64; t0 = T;
  o4 = __builtin_addcl(o4, t0, 0, &c0);
  tt = t1;
  T = (u128)b[4]*a[1]; t1 = T>>64; t0 = T;
  t0 = __builtin_addcl(tt, t0, 0, &c1);
  o5 = __builtin_addcl(o5, t0, c0, &c0);
  t1 = __builtin_addcl(0, t1, c1, &c1);
  o6 = __builtin_addcl(0, t1, c0, &c0);

  o[0] = o0;
  o[1] = o1;
  o[2] = o2;
  o[3] = o3;
  o[4] = o4;
  o[5] = o5;
  o[6] = o6;
}

static void mu8u6u2(u64 *o, const u64 *b, const u64 *a){
  u128 T; u64 t0,t1,tt,o0,o1,o2,o3,o4,o5,o6,o7,c,c0,c1;
  T = (u128)b[0]*a[0]; t1 = T>>64; t0 = T;
  o0 = t0;
  tt = t1;
  T = (u128)b[0]*a[1]; t1 = T>>64; t0 = T;
  o1 = __builtin_addcl(tt, t0, 0, &c);
  o2 = __builtin_addcl(0, t1, c, &c);

  T = (u128)b[1]*a[0]; t1 = T>>64; t0 = T;
  o1 = __builtin_addcl(o1, t0, 0, &c0);
  tt = t1;
  T = (u128)b[1]*a[1]; t1 = T>>64; t0 = T;
  t0 = __builtin_addcl(tt, t0, 0, &c1);
  o2 = __builtin_addcl(o2, t0, c0, &c0);
  t1 = __builtin_addcl(0, t1, c1, &c1);
  o3 = __builtin_addcl(0, t1, c0, &c0);

  T = (u128)b[2]*a[0]; t1 = T>>64; t0 = T;
  o2 = __builtin_addcl(o2, t0, 0, &c0);
  tt = t1;
  T = (u128)b[2]*a[1]; t1 = T>>64; t0 = T;
  t0 = __builtin_addcl(tt, t0, 0, &c1);
  o3 = __builtin_addcl(o3, t0, c0, &c0);
  t1 = __builtin_addcl(0, t1, c1, &c1);
  o4 = __builtin_addcl(0, t1, c0, &c0);

  T = (u128)b[3]*a[0]; t1 = T>>64; t0 = T;
  o3 = __builtin_addcl(o3, t0, 0, &c0);
  tt = t1;
  T = (u128)b[3]*a[1]; t1 = T>>64; t0 = T;
  t0 = __builtin_addcl(tt, t0, 0, &c1);
  o4 = __builtin_addcl(o4, t0, c0, &c0);
  t1 = __builtin_addcl(0, t1, c1, &c1);
  o5 = __builtin_addcl(0, t1, c0, &c0);

  T = (u128)b[4]*a[0]; t1 = T>>64; t0 = T;
  o4 = __builtin_addcl(o4, t0, 0, &c0);
  tt = t1;
  T = (u128)b[4]*a[1]; t1 = T>>64; t0 = T;
  t0 = __builtin_addcl(tt, t0, 0, &c1);
  o5 = __builtin_addcl(o5, t0, c0, &c0);
  t1 = __builtin_addcl(0, t1, c1, &c1);
  o6 = __builtin_addcl(0, t1, c0, &c0);

  T = (u128)b[5]*a[0]; t1 = T>>64; t0 = T;
  o5 = __builtin_addcl(o5, t0, 0, &c0);
  tt = t1;
  T = (u128)b[5]*a[1]; t1 = T>>64; t0 = T;
  t0 = __builtin_addcl(tt, t0, 0, &c1);
  o6 = __builtin_addcl(o6, t0, c0, &c0);
  t1 = __builtin_addcl(0, t1, c1, &c1);
  o7 = __builtin_addcl(0, t1, c0, &c0);

  o[0] = o0;
  o[1] = o1;
  o[2] = o2;
  o[3] = o3;
  o[4] = o4;
  o[5] = o5;
  o[6] = o6;
  o[7] = o7;
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

static void sarn(int n, u64 *a, int k){
  if(__builtin_expect(k<0, 0)){
    shln(n,a,-k);
  } else {
    int off = k>>6;
    u64 *src = a + off, *dst = a, *aend = a + n;
    i64 c;
    if(__builtin_expect(src < aend, 1)){
      int s = k&63, q = ~k&63;
      c = *src++;
      while(__builtin_expect(src < aend, 1)){
	u64 nt = *src++;
	*dst++ = (u64)c>>s|nt<<1<<q;
	c = nt;
      }
      *dst++ = c>>s;
    } else {
      c = a[n-1];
    }
    c >>= 63;
    while(__builtin_expect(dst < aend, 0)) *dst++ = c;
  }
}

static inline u128 uq(const u64 *c){
  return (u128)c[1]<<64|c[0];
}

// signed 3 64-bit limb square
static inline void sqru6i3(u6x64 o, const u3x64 a){
  u64 c0, b0, b1, b2, b3, t0, t1, t2;
  u128 a2a2 = (u128)a[2]*a[2];
  u128 a2a1 = (u128)a[2]*a[1];
  u128 a2a0 = (u128)a[2]*a[0];
  u128 a1a1 = (u128)a[1]*a[1];
  u128 a1a0 = (u128)a[1]*a[0];
  u128 a0a0 = (u128)a[0]*a[0];
  t2 = a[2]<<1|a[1]>>63;
  t1 = a[1]<<1|a[0]>>63;
  t0 = a[0]<<1;
  i64 m = a[2]; m >>= 63;
  t2 &= m;
  t1 &= m;
  t0 &= m;
  o[0] = a0a0;
  o[1] = a0a0>>64;
  o[2] = a1a1;
  o[3] = a1a1>>64;
  o[4] = a2a2;
  o[5] = a2a2>>64;
  b0  = a1a0;
  b1  = __builtin_addcl(a2a0, a1a0>>64,  0, &c0);
  b2  = __builtin_addcl(a2a1, a2a0>>64, c0, &c0);
  b3  = __builtin_addcl(   0, a2a1>>64, c0, &c0);
  o[1] = __builtin_addcl(o[1], b0,  0, &c0);
  o[2] = __builtin_addcl(o[2], b1, c0, &c0);
  o[3] = __builtin_addcl(o[3], b2, c0, &c0);
  o[4] = __builtin_addcl(o[4], b3, c0, &c0);
  o[5] = __builtin_addcl(o[5],  0, c0, &c0);
  o[1] = __builtin_addcl(o[1], b0,  0, &c0);
  o[2] = __builtin_addcl(o[2], b1, c0, &c0);
  o[3] = __builtin_addcl(o[3], b2, c0, &c0);
  o[4] = __builtin_addcl(o[4], b3, c0, &c0);
  o[5] = __builtin_addcl(o[5],  0, c0, &c0);
  o[3] = __builtin_subcl(o[3], t0, c0, &c0);
  o[4] = __builtin_subcl(o[4], t1, c0, &c0);
  o[5] = __builtin_subcl(o[5], t2, c0, &c0);
}

// unsigned 6 64-bit limb approximate square
static inline void sqrhu6(u6x64 o, const u6x64 a){
  u64 c0, o0, o1, o2, o3, o4, o5;
  u128 a3a2 = (u128)a[3]*a[2];
  u128 a4a1 = (u128)a[4]*a[1];
  o0  = __builtin_addcl(a3a2>>64, a4a1>>64,   0, &c0);
  o1  = __builtin_addcl(       0,        0,  c0, &c0);
  u128 a5a0 = (u128)a[5]*a[0];
  o0  = __builtin_addcl(      o0, a5a0>>64,   0, &c0);
  o1  = __builtin_addcl(      o1,        0,  c0, &c0);

  u128 a4a2 = (u128)a[4]*a[2];
  o0  = __builtin_addcl(      o0,     a4a2,   0, &c0);
  o1  = __builtin_addcl(      o1, a4a2>>64,  c0, &c0);
  o2  = __builtin_addcl(       0,        0,  c0, &c0);

  u128 a5a1 = (u128)a[5]*a[1];
  o0  = __builtin_addcl(      o0,     a5a1,   0, &c0);
  o1  = __builtin_addcl(      o1, a5a1>>64,  c0, &c0);
  o2  = __builtin_addcl(      o2,        0,  c0, &c0);

  u128 a4a3 = (u128)a[4]*a[3];
  o1  = __builtin_addcl(      o1,     a4a3,   0, &c0);
  o2  = __builtin_addcl(      o2, a4a3>>64,  c0, &c0);
  o3  = __builtin_addcl(       0,        0,  c0, &c0);

  u128 a5a2 = (u128)a[5]*a[2];
  o1  = __builtin_addcl(      o1,     a5a2,   0, &c0);
  o2  = __builtin_addcl(      o2, a5a2>>64,  c0, &c0);
  o3  = __builtin_addcl(      o3,        0,  c0, &c0);

  u128 a5a3 = (u128)a[5]*a[3];
  o2  = __builtin_addcl(      o2,     a5a3,   0, &c0);
  o3  = __builtin_addcl(      o3, a5a3>>64,  c0, &c0);
  o4  = __builtin_addcl(       0,        0,  c0, &c0);

  u128 a5a4 = (u128)a[5]*a[4];
  o3  = __builtin_addcl(      o3,     a5a4,   0, &c0);
  o4  = __builtin_addcl(      o4, a5a4>>64,  c0, &c0);
  o5  = __builtin_addcl(       0,        0,  c0, &c0);

  o0  = __builtin_addcl(o0, o0,  0, &c0);
  o1  = __builtin_addcl(o1, o1, c0, &c0);
  o2  = __builtin_addcl(o2, o2, c0, &c0);
  o3  = __builtin_addcl(o3, o3, c0, &c0);
  o4  = __builtin_addcl(o4, o4, c0, &c0);
  o5  = __builtin_addcl(o5, o5, c0, &c0);

  u128 a3a3 = (u128)a[3]*a[3];
  o[0]  = __builtin_addcl(o0,     a3a3,  0, &c0);
  o[1]  = __builtin_addcl(o1, a3a3>>64, c0, &c0);
  u128 a4a4 = (u128)a[4]*a[4];
  o[2]  = __builtin_addcl(o2,     a4a4, c0, &c0);
  o[3]  = __builtin_addcl(o3, a4a4>>64, c0, &c0);
  u128 a5a5 = (u128)a[5]*a[5];
  o[4]  = __builtin_addcl(o4,     a5a5, c0, &c0);
  o[5]  = __builtin_addcl(o5, a5a5>>64, c0, &c0);
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

static inline u128 reciprocalU(u128 kd){
  u128 kdh = kd>>64, kdl = (u64)kd;
  u128 n = (u128)1<<127;
  u64 r = n/kdh;
  if(__builtin_expect(!r,0)) r = ~0ull;
  i128 Hh = kdh*r; u128 Hl = kdl*r;
  Hh += Hl>>64;
  Hh <<= 57;
  Hh |= (u64)Hl>>7;
  i64 hh = Hh>>92;
  i128 dR = mhuU(r, Hh) - ((i128)((hh>>63)&r)<<64);
  dR >>= 56;
  u128 R = r; R<<=64;
  R -= dR;
  return R;
}

// 0 - ordinary number, 1 -- infinity, 2 -- snan, 3 -- qnan
static inline char getclass(u128 x){
  u64 xh = x>>64, xl = x;
  int t = xh>>32 | !!(xh<<32|xl);
  return (t>=0x7fff<<16) + (t>=(0x7fff<<16)+1) + (t>=(0x7fff8<<12));
}

static const char indl[] = {
  0, 0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 8, 8, 9, 10, 10, 11, 12, 12,
  13, 13, 14, 15, 15, 16, 16, 17, 18, 18, 19, 19, 20, 21, 21, 22, 22,
  23, 24, 24, 25, 25, 26, 26, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32,
  33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40, 40, 41,
  41, 42, 42, 43, 43, 44, 44, 45, 45, 45, 46, 46, 47, 47, 48, 48, 49,
  49, 49, 50, 50, 51, 51, 52, 52, 52, 53, 53, 54, 54, 54, 55, 55, 55,
  56, 56, 57, 57, 57, 58, 58, 58, 59, 59, 59, 60, 60, 61, 61, 61, 62,
  62, 62, 63, 63, 63,
};
static const char indh[] = {
  0, 0, 1, 1, 2, 2, 3, 4, 5, 5, 5, 6, 6, 7, 8, 9, 10, 10, 10, 11, 11,
  11, 12, 12, 13, 14, 14, 15, 16, 16, 17, 18, 19, 20, 20, 20, 21, 21,
  21, 22, 22, 23, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 29, 29,
  30, 31, 31, 32, 33, 34, 34, 35, 36, 37, 38, 38, 38, 38, 39, 39, 39,
  39, 40, 40, 40, 41, 41, 41, 41, 42, 42, 42, 43, 43, 43, 44, 44, 44,
  45, 45, 46, 46, 46, 47, 47, 47, 48, 48, 49, 49, 49, 50, 50, 51, 51,
  52, 52, 53, 53, 54, 54, 54, 55, 55, 56, 57, 57, 58, 58, 59, 59, 60,
  60, 61, 62, 62, 63
};
// a 15 bit approximation of tan(i*pi/4/64)
static const unsigned short tn[] = {
  0, 0x192, 0x324, 0x4b7, 0x64a, 0x7dd, 0x971, 0xb06, 0xc9b, 0xe32, // 10
  0xfca, 0x1162, 0x12fd, 0x1498, 0x1636, 0x17d5, 0x1976, 0x1b19, // 18
  0x1cbe, 0x1e66, 0x2010, 0x21bd, 0x236c, 0x251f, 0x26d4, 0x288d, // 26
  0x2a49, 0x2c09, 0x2dcd, 0x2f94, 0x3160, 0x3330, 0x3505, 0x36de, // 34
  0x38bd, 0x3aa1, 0x3c8a, 0x3e79, 0x406e, 0x4269, 0x446b, 0x4673, // 42
  0x4883, 0x4a9a, 0x4cb8, 0x4edf, 0x510e, 0x5346, 0x5587, 0x57d1, // 50
  0x5a26, 0x5c85, 0x5eee, 0x6163, 0x63e4, 0x6672, 0x690c, 0x6bb4, // 58
  0x6e6a, 0x712f, 0x7403, 0x76e8, 0x79de, 0x7ce5, 0x8000};
// phi0[i] = atan(tn[i]);
static const u6x64 phi0[] = {
  {0, 0, 0, 0, 0, 0},
  {0x02c1e09886f8fc65, 0x12befe0e684801ab, 0x5b500d3aa545cf10, 0xc29c938f5e9594c8, 0xe19fb2a3207f6303, 0x00c8fd6b34169a4d},
  {0x4cb54df37491326d, 0xbcc2d6b5024de6ba, 0x0140a83479a7d09b, 0xb47181e24638fe92, 0x94e685c329b2d773, 0x0191eb5b0f2af2cb},
  {0x96290951ce6369c6, 0xe6f739fc68793759, 0x23222e343d6f817c, 0xb5a31899b4d03cbb, 0xe18dc8765eb3b239, 0x025b3a2f01f4cf05},
  {0x26a8b16ac189860c, 0xec201bca4b528ce4, 0xd8b160e7c9c348ed, 0xf7384a1bcde479ed, 0x3665e6a2207124bc, 0x03245a68934296f4},
  {0x2b2747a2b7b246bd, 0x0eb316ea6af2c481, 0x8994c96123649fad, 0x40c4923addf07637, 0x7eb0104c3bbaeebb, 0x03ed3c99ec5f4fd6},
  {0xc1aa40ed9cf81137, 0x7b9341c26981ba2c, 0xa3f94febb560fec3, 0x038b8f349e0b9941, 0x60a39999b30013e3, 0x04b650c091b47191},
  {0x23fb4c9fabd5a3f5, 0x8e9c1d3e3a2a9e80, 0x4f7d972251777188, 0x7f688f435290217b, 0xa760b14ee5c8185b, 0x057f86ef8cce05d9},
  {0xa345983253154b9f, 0x742e9e3f6bbf6db1, 0xbe3b949d708e4ab7, 0xb8f721b926dd7968, 0xbc9cc590f04c39b1, 0x0648506548567e1c},
  {0x4270e6ec850ed3d4, 0xbd6add47dd4704ae, 0xd7f4f0121bbb10d8, 0x6420acc6d7aa717a, 0x5181cc2b7f67466d, 0x07119afb1db64d3b},
  {0x3c0ddcb933c0132f, 0x29ac0726e281c9c4, 0x8af040655ae13a00, 0xe5e69a461efee36f, 0x98430538072960bc, 0x07dad79d66d396e1},
  {0xc3d6eaaa86b1e257, 0x478e9831c107b273, 0x91ece4ca7c8c9a92, 0x7e4d9cd1133d108e, 0x6b2499055ef359f2, 0x08a378a00557419c},
  {0xe452b1a89a5919ea, 0x3cfb3638e459c56f, 0x82a58ff9c80f179f, 0x98506bfc8617b00d, 0xeef4781de96151b3, 0x096ce71f84443705},
  {0x7022ba3fd2c3a3ec, 0x1e810ce3f03023fb, 0x080ea2ce1b951a1b, 0xdc28825e93081751, 0xfa554a296e7bbd3e, 0x0a359a37ad347506},
  {0x5082480b4642639f, 0xf6d143858afffc0b, 0xe83a1d095ceb9d93, 0xf1c6b6f3f95d034c, 0xbe30339b35470763, 0x0afef860ced6731a},
  {0x06a1328f3ed37ce1, 0xa8fbddade1ab0e4e, 0xace3bbf391d4780d, 0xc419de3fafb1443c, 0x71d3f327dc1666e5, 0x0bc7f750a98aaf11},
  {0x2cdef1893d2be331, 0x1644a02bce301472, 0xe8835bd11f8a7463, 0x7f78b6932fe4c9a9, 0x573436e5491c3c3b, 0x0c910288a5912f73},
  {0x820f7fdd710ebd56, 0xba60aab851afd713, 0x4453d264039e6104, 0xd07a8bd375db11fb, 0x1a70ad3ce98dd37d, 0x0d5a08c366f7ad26},
  {0x23c5850e662df2a8, 0x48eb29d7959af0d9, 0x1db3e8ca092b26ff, 0x1576eaf97ffde937, 0x2b63c532aae1baef, 0x0e22f8c260ccbeb2},
  {0x5a1cbdcc77ccbebd, 0x42d39881973ab439, 0xea1ab61b1291bf62, 0x61d834c4d89794de, 0x3ea9630254b006a7, 0x0eec3a81159ad32d},
  {0x354baa05f6e10d1c, 0x2ff82bd8a784b660, 0x2161dc75b94d9dad, 0xf2cebf4c84ed043f, 0xf9f105fb8919b427, 0x0fb5424b656558f6},
  {0xb89dde65241d6464, 0xbb4ad1f58d638111, 0xfa29639aeb3ba102, 0x9a4c65cd7e32af7d, 0x7ed90db179691680, 0x107e76c03c9dea1c},
  {0xc92938551b12cad8, 0xec7dcfe04738fefb, 0x2937dd301a0b4ca0, 0x01f63aedeb6b1188, 0x3e1132674a1a6ece, 0x11474db47224d179},
  {0xb8cd0fc44f74d861, 0x2e822666fd0d636e, 0x8e1d4cec346d99be, 0xc6330a710a48d5da, 0x5f1a2b16680c1a30, 0x1210a27e5ff90c02},
  {0x5e1c793da47d6569, 0xfff7d2b2957a644f, 0xeeebd0bcf6fee6c2, 0xeb7eb29db9b819fc, 0xe00180e9012b321a, 0x12d974fae8ccb2fc},
  {0x8a7aa5432dddb0a8, 0xf6f9c2e4f0d2b4b7, 0x2c27002eeef60e9f, 0xe93934b45f17a029, 0x10ab1e3995f97abe, 0x13a29d51b33d21e1},
  {0x8016d96ec5b3e5df, 0x0d12ea7481262996, 0x0ba014f4fcf82814, 0xa94b6c896608a877, 0x0ce9531d42e1282f, 0x146b92651a66881b},
  {0xa3a7dc6a86bc6ca7, 0x6832f1af6eeb7d1b, 0x80fa69802980962a, 0x4fbf24d0d8a67b5d, 0x81c083440206ac79, 0x1534b3c5ac621e6d},
  {0x10236fc7bc489479, 0xe866bbcf232064e3, 0x8decedc977f53a0c, 0x01533415a5b2003d, 0xf33cb2ee84283a29, 0x15fdebcff69fce1e},
  {0x1c6a16ed73097fed, 0x04b6b552132a277c, 0x873e480ede4b49f9, 0xd2e10ecf59b39d8d, 0x3181a4ff744e39c8, 0x16c6b4791c7551cb},
  {0x3762db8a2f46ae6c, 0xe605126135c4d13e, 0x95cd248f2199c031, 0x046e6925e63b921f, 0x1edbbe0a60738519, 0x178fda3cf9a1f86a},
  {0x622dd7b0d6052b2b, 0x6c52c04b49a43f67, 0x88c53f408d6effd6, 0x2f9443c296486b1e, 0xb29d416f839b4c6a, 0x1858d64de0abafb9},
  {0x41e74e55d8377c5c, 0x6d71bf109d52b4b4, 0x257cc4f9c5ac1896, 0xb36f178c4ec3381d, 0x48b86fb55342a616, 0x192200ca656f9b4e},
  {0x9ce847bba8a998f8, 0xd11a4c5b5dd0f414, 0xe01bcb3136d1ebdb, 0xa0862f030b9c517c, 0xa825b4e5ba19aa06, 0x19ead54cc2ca2709},
  {0x414da7fe923aa6b3, 0x7937289ae91c4e1e, 0x023d066c6e855b6c, 0xd806ede78bcf8603, 0x7557aecadb39e171, 0x1ab4151b72c17c74},
  {0xeba76a289fd187b2, 0x806ebe30e6895092, 0xc8828c04732a6e15, 0x7160eba8b1434d38, 0x425d9aff030711ea, 0x1b7d3b004cb73b1d},
  {0xe3de9fa9dfe3af41, 0x88b2384a90ac9a3c, 0x4e49b80c84414a77, 0xc31388ba77526296, 0xadf256ec30dee5f4, 0x1c462f4336d5d87f},
  {0xc8016db77a6fe180, 0x74d367ad0de559f3, 0x3590c43f99974721, 0xde083347f4ae4c39, 0x83520693c00f13ae, 0x1d0f41d64261e9ad},
  {0x90d94a7879b7f974, 0x7f4261801e9d8880, 0xe65deec790b85094, 0xc1610e738a4c9f6c, 0xc4c9618e94bf1b41, 0x1dd857e3ab6ae652},
  {0xcf5ca650e7803992, 0xda4170d655e373a3, 0xd03830653bb48d42, 0x389df7ea5a405f86, 0x988fa1d2cc3a0516, 0x1ea156d864e87301},
  {0x63ac43b79fd4a119, 0x860223d73b8c2c2c, 0xd57b1b3b47b73093, 0x19cdea35fab86a95, 0xa4546bf333ad5f5e, 0x1f6a88026bdc2f47},
  {0x1732b191c4b0e778, 0x9352dbb5661dd274, 0x144fe9b20ee58259, 0x0eb5ed9afee849a5, 0xad9b43ad3c3e1fa7, 0x20336b52f6bb9ab6},
  {0xa982314f96806c98, 0x0b0c5b3f526aa38c, 0x90f83830b407e311, 0xad1715d205e8ce0e, 0x196bdc15af2356b3, 0x20fca91c686855d2},
  {0xfe926dfb72a70556, 0x3b5f2a42166f3183, 0xa62b33bca0f85eca, 0xed2dbb498fb85d4d, 0xf309e31fbdc9aeb7, 0x21c5c0b58968023e},
  {0xcf2fe2ca74009c66, 0xa73a2906aba43e6e, 0x9871f53ec824467b, 0x1a38ceab9fce7f8f, 0xbfdeff67b18287c1, 0x228e9579f321c783},
  {0x4d3b3149caa4f593, 0xa46a448c68bdefa9, 0xaa62fe638979d0b3, 0x56d7577e496786b1, 0x9f94db3d59a078d7, 0x2357c4da5035f70f},
  {0xf4eabdca73374af6, 0xae0684761c9ebb0f, 0x44cdeed621e49a28, 0x7928137ddd0fa064, 0xfd755dd3bdcbab23, 0x2420cf7ee51f46e0},
  {0xd104943294266a7e, 0xa603aaff8ea1b4e0, 0xf074b304a436b2e8, 0x08f6bac958fc93e0, 0x6402f3a15aabb8f2, 0x24e9f01e8e1f712c},
  {0x3d3b9d056aa1ee79, 0x345a09eb2405f477, 0xe168c8e2fb055c2a, 0x49420a6b6791619d, 0x8787fb8a5a45b05b, 0x25b303d3106131a6},
  {0xb805175699cb99a9, 0x4fe4be06b00aec48, 0xd121ef2ed145dbc9, 0x6d83a3084993fdc2, 0xcd1219f2665f60d6, 0x267be85a925b38d2},
  {0x73b42b0d4b9f4cf9, 0xa7b9b20992b3d8a1, 0x558f26417137e904, 0x257de5e0913d9714, 0x13b1717f1dc3721f, 0x2745274cfd47c28c},
  {0x2c961ad29e10d590, 0xbe94571f4b227c15, 0xc65400b7a5517ca5, 0xf8c749addac40559, 0xd2adc89b03c20921, 0x280e42f64d3f9844},
  {0x279b685af05d793c, 0xddfa01d66dc021d9, 0x03e0fc67ab913ee1, 0x6c0a3ecf5778f283, 0xe8e9142622ebdbb4, 0x28d716f50d9e9584},
  {0xcf4ec61ec1887d94, 0x502713a07a267307, 0x5fdbb5a9b9c8750a, 0xd5cf19043439b62c, 0x1136bf78715f5851, 0x29a021f7d8cb387a},
  {0xbca0a7dd978cef70, 0x98e574273b015563, 0x67a77b57da9afb00, 0x54de998591fc535c, 0xa86e6b5b7a9bae15, 0x2a693870fc67c444},
  {0x7c6cbe72316ce0d5, 0x1e0fd83317165e74, 0x71adf16aa60f3af3, 0xa9341d97e4f6adcb, 0xe717978b9134d491, 0x2b327dd7ab2ca09d},
  {0xa0ae2c41858d5c87, 0x239d87b916e40920, 0xa024d4fa90f4f26b, 0xb55be89ce49ef928, 0x28b56505dd97d3dd, 0x2bfb77a035071fb6},
  {0xb0030b87c9b62d18, 0x9992bf2e20ce7cd4, 0x1765547834a268e5, 0xa187fec213be1016, 0x3fbcd166548791ee, 0x2cc4936b1987b1b3},
  {0xd339bea7ca0b74e0, 0xc29c5e9826b1f6e3, 0x2f5e5ab61673e3f6, 0xff998a6f2d19532f, 0x9d5322247bb149e5, 0x2d8da0fcd932a561},
  {0xd20d77dc0b077313, 0xbfccaa053a043a9b, 0x12b3010be6a1eb3f, 0xf316fcb46d0ffb40, 0x4095f9630a4eb3e1, 0x2e56b94da0515984},
  {0xd168195d379149a7, 0x60a53f3de0e7efd3, 0x3658cd3898020ce5, 0xa30583591a2dffbe, 0x1e375bd40e0e3933, 0x2f1faa4f3a17939d},
  {0xee927d556dfb2ae8, 0xb5cd6ae1598c854e, 0x217441a97323d25e, 0xe696c043bb881630, 0xe2445ad229e46e95, 0x2fe8ccf5ebf0a133},
  {0x159fdfc634d4d38a, 0x9724055712324ced, 0x7c59984442c9d80a, 0x6b5f6d06b34c1da6, 0x12b2f428ff7acb76, 0x30b1e91cf0943362},
  {0xa9acfc1b07dbdc4d, 0xcc77842a2fe6f682, 0xe7f3a1e83c315272, 0x19506e1283ee2a6b, 0xbe60466777b9622d, 0x317ac874b0fba710},
};

static const u6x64 pio2 = 
  {0xf7ca8cd9e69d218e, 0x28a5043cc71a026e, 0x105df531d89cd91, 0x948127044533e63a, 0x62633145c06e0e68, 0x6487ed5110b4611a};

static __float128 as_atanq_accurate(__float128);
__float128 atanq(__float128);

__float128 cr_atanq(__float128 x) {
  unsigned flagp = _mm_getcsr(), oflagp = flagp, rm = flagp&_MM_ROUND_MASK;
  const u64 smsk = 1ull<<63, inf = 0x7fffull<<48;
  b128u128_u X = {.a = reinterpret_f128_as_u128(x)};
  u64 xsgn = X.b[1]&smsk;
  X.b[1] &= ~smsk; // strip sign
  int xn = X.b[1]>>48;
  // for |x|<=0x1.d12ed0af1a27ef3dabae134c2da3p-57 the difference atan(x)-x is less than 0.5 ulp
  if(__builtin_expect(xn<0x3fff-57,0)){ // |x|<=0x1p-58
    if(X.a == 0) return x; // x=+/-0 
    X.a -= (rm != _MM_ROUND_NEAREST)*(!!xsgn*(rm==_MM_ROUND_UP) + !xsgn*(rm==_MM_ROUND_DOWN) + (rm==_MM_ROUND_TOWARD_ZERO));
    if(X.b[1] < 1ull<<48){
      flagp |= FE_UNDERFLOW;
#ifdef CORE_MATH_SUPPORT_ERRNO
      errno = ERANGE; // underflow
#endif
    }
    X.b[1] |= xsgn;
    flagp |= FE_INEXACT;
    // set inexact flag only if it is not set before
    if(__builtin_expect(oflagp!=flagp, 0)) _mm_setcsr(flagp);
    __float128 res = reinterpret_u128_as_f128(X.a); // put into xmm register
    return res;
  }
  if(__builtin_expect(xn>0x3fff+114,0)){ // |x|>=0x1p115, Inf, NaN
    char xnan = getclass(X.a);
    if(xnan==2){ // signaling NAN
      flagp |= FE_INVALID;
      if(__builtin_expect(oflagp!=flagp, 0)) _mm_setcsr(flagp);
#if (defined(_WIN32) || defined(__APPLE__))
      return 0.0q / 0.0q;
#else
      return __builtin_nanf128("atanq");
#endif
    } else if(xnan==3) {//quiet NAN
      return x; // propagate nan
    }
    X.f = 0x1.921fb54442d18469898cc51701b8p+0q;
    X.a += (rm != _MM_ROUND_NEAREST)*(!xsgn*(rm==_MM_ROUND_UP) + !!xsgn*(rm==_MM_ROUND_DOWN));
    X.b[1] |= xsgn;
    flagp |= FE_INEXACT;
    // set inexact flag only if it is not set before
    if(__builtin_expect(oflagp!=flagp, 0)) _mm_setcsr(flagp);
    __float128 res = reinterpret_u128_as_f128(X.a); // put into xmm register
    return res;
  }
  X.b[1] &= ~inf;
  X.b[1] |= 1ull<<48;

  u128 kn, kd;
  int dn;
  long isct, g = xn>=0x3fff;
  if(!g){
    int e = 0x3fff - xn;
    if(__builtin_expect(e>7, 0)){
      isct = 0;
      kn = X.a<<15;
      kd = (u128)1<<127;
      dn = e;
    } else {
      isct = indl[X.b[1] >> (e+41)];
      isct += tn[isct+1] <= (X.b[1]>>(e+33));
      u64 tt = tn[isct];
      tt <<= 33 + e;
      u64 knh = X.b[1] - tt, knl = X.b[0];
      int nzn;
      if(__builtin_expect(knh, 1)){
	nzn = __builtin_clzll(knh);
	knh = knh<<nzn|knl>>(-nzn&63);
	knl = knl<<nzn;
	kn = (u128)knh<<64|knl;
      } else {
	nzn = __builtin_clzll(knl);
	knl <<= nzn;
	kn = (u128)knl<<64;
	nzn += 64;
      }
      X.a *= tn[isct];
      X.b[0] = X.b[0]>>e|X.b[1]<<(-e&63);
      X.b[1] = X.b[1]>>e|1ull<<63;
      kd = X.a;
      dn = e + nzn - 15;
    }
  } else {
    int e = xn - 0x3fff;
    if(__builtin_expect(e>6, 0)){
      isct = 0;
      kn = (u128)1<<127;
      kd = X.a<<15;
      dn = e;
    } else {
      isct = indh[(X.b[1]^(~0ull>>16)) >> (e+42)];
      isct += tn[isct+1]*X.b[1] < (1ull<<(63-e));
      b128u128_u t = {.a = X.a * tn[isct]};
      if(__builtin_expect(t.b[1]&(1ull<<(63-e)), 0))
      	t.a = X.a * tn[--isct];
      t.b[1] <<= e;
      if(e) t.b[1] |= t.b[0]>>(-e&63);
      t.b[0] <<= e;
      kn = -t.a;
      u64 knh = kn>>64, knl = kn;
      knh ^= 1ull<<63;
      int nzn;
      if(__builtin_expect(knh, 1)){
	nzn = __builtin_clzll(knh);
	knh = knh<<nzn|knl>>(-nzn&63);
	knl = knl<<nzn;
	kn = (u128)knh<<64|knl;
      } else {
	nzn = __builtin_clzll(knl);
	knl <<= nzn;
	kn = (u128)knl<<64;
	nzn += 64;
      }
      u64 tt = tn[isct];
      tt <<= 33 - e;
      X.b[1] += tt;
      int nzd = __builtin_clzll(X.b[1]);
      X.b[1] = X.b[1]<<nzd|X.b[0]>>(-nzd&63);
      X.b[0] = X.b[0]<<nzd;
      kd = X.a;
      dn = e + nzn - nzd + 15;
    }
  }
  u128 R = reciprocalU(kd), T = mhUU(kn,R), T2 = sqrhU(T);
  int dn2 = 2*dn-12;
  if(__builtin_expect(dn2<128, 1))
    T2 >>= dn2;
  else
    T2 = 0;
  
  static const u64 c[][2] = {
    {0xffffffffffffffffull, 0xffffffffffffffffull},
    {0x555555555555554full, 0x0015555555555555ull},
    {0x3333333333332f55ull, 0x0000033333333333ull},
    {0x49249249249147dfull, 0x0000000092492492ull},
    {0x1c71c71c71a599e3ull, 0x00000000001c71c7ull},
    {0x745d1745cf00b15full, 0x00000000000005d1ull},
    {0x3b13b13af8b70dd8ull, 0x0000000000000001ull},
    {0x004444439744d102ull, 0x0000000000000000ull},
    {0x00000f0cb8c66f08ull, 0x0000000000000000ull}
  };

  u64 t2h = T2>>64, fl = c[8][0];
  fl = c[7][0] - mhuu(t2h, fl);
  fl = c[6][0] - mhuu(t2h, fl);
  u128 f = (u128)c[5][1]<<64|(c[5][0]-t2h-mhuu(t2h, fl));
  int i = 5;
  while(--i >= 1) f = uq(c[i]) - mhUU(T2,f);
  f = mhUU(T2, f);
  f = mhUU(T, f);
  f = T - f;
  b128u128_u v, dv;
  u64 rnd;
  if(__builtin_expect(isct|g,1)){
    u3x64 f3 = {0,f,f>>64};
    ++dn;
    if(__builtin_expect(dn<64, 1)){
      f3[0] =           f3[1]<<(64-dn);
      f3[1] = f3[1]>>dn|f3[2]<<(64-dn);
      f3[2] = f3[2]>>dn;
    } else
      shrn(3,f3,dn);
    addu3u3u3(f3,phi0[isct]+3,f3);
    if(g) subu3u3u3(f3, pio2+3, f3);
    int k = __builtin_clzll(f3[2]);
    rnd = (f3[1]>>(14-k))&1;
    u128 t = (u128)f3[1]<<64|f3[0];
    const u64 eps = 0xca2339c0ebedfa4ull;
    t += eps;
    u64 th = t>>64, tl = t;
    th &= (1ull<<(15-k))-1;
    th ^= (u64)(rm == _MM_ROUND_NEAREST)<<(14-k);
    if(th==0 && tl<0x194467381d7dbf48ull) return as_atanq_accurate(x);
    xn = 0x3fff - k;
    v.b[0] = f3[1]>>(15-k)|f3[2]<<(49+k);
    v.b[1] = f3[2]>>(15-k);
  } else {
    v.a = f;
    int k = __builtin_clzll(v.b[1]);
    xn = 0x3ffe - dn - k;
    u64 tl = (v.b[0] + 6) & (~0ull>>(49+k));
    tl ^= (u64)(rm == _MM_ROUND_NEAREST)<<(14-k);
    if(tl<=15) return as_atanq_accurate(x);
    rnd = (v.b[0]>>(14-k))&1;
    v.b[0] = v.b[0]>>(15-k)|v.b[1]<<(49+k);
    v.b[1] = v.b[1]>>(15-k);
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

__float128 as_atanq_accurate(__float128 x){  
  unsigned flagp = _mm_getcsr(), oflagp = flagp, rm = flagp&_MM_ROUND_MASK;
  const u64 smsk = 1ull<<63, inf = 0x7fffull<<48;
  b128u128_u X = {.a = reinterpret_f128_as_u128(x)};
  u64 xsgn = X.b[1]&smsk;
  X.b[1] &= ~smsk; // strip sign
  int xn = X.b[1]>>48;
  X.b[1] &= ~inf;
  X.b[1] |= 1ull<<48;

  u128 kn;
  u3x64 kd;
  int dn;
  long isct, g = xn>=0x3fff;
  if(!g){
    int e = 0x3fff - xn;
    if(e>7){
      isct = 0;
      kn = X.a<<15;
      kd[2] = 1ull<<63;
      kd[1] = 0;
      kd[0] = 0;
      dn = e;
    } else {
      isct = indl[X.b[1] >> (e+41)];
      isct += tn[isct+1] <= (X.b[1]>>(e+33));
      u64 tt = tn[isct];
      tt <<= 33 + e;
      u64 knh = X.b[1] - tt, knl = X.b[0];
      int nzn;
      if(__builtin_expect(knh, 1)){
	nzn = __builtin_clzll(knh);
	knh = knh<<nzn|knl>>(-nzn&63);
	knl = knl<<nzn;
	kn = (u128)knh<<64|knl;
      } else {
	nzn = __builtin_clzll(knl);
	knl <<= nzn;
	kn = (u128)knl<<64;
	nzn += 64;
      }
      X.a *= tn[isct];
      kd[2] = X.b[1];
      kd[1] = X.b[0];
      kd[0] = 0;
      if(e){
	kd[0] =          kd[1]<<(-e&63);
	kd[1] = kd[1]>>e|kd[2]<<(-e&63);
	kd[2] = kd[2]>>e;
      }
      kd[2] |= 1ull<<63;
      dn = e + nzn - 15;
    }
  } else {
    int e = xn - 0x3fff;
    if(e>6){
      isct = 0;
      kn = (u128)1<<127;
      X.a <<= 15;
      kd[2] = X.b[1];
      kd[1] = X.b[0];
      kd[0] = 0;
      dn = e;
    } else {
      isct = indh[(X.b[1]^(~0ull>>16)) >> (e+42)];
      isct += tn[isct+1]*X.b[1] < (1ull<<(63-e));
      b128u128_u t = {.a = X.a * tn[isct]};
      if(__builtin_expect(t.b[1]&(1ull<<(63-e)), 0))
      	t.a = X.a * tn[--isct];
      t.b[1] <<= e;
      if(e) t.b[1] |= t.b[0]>>(-e&63);
      t.b[0] <<= e;
      kn = -t.a;
      u64 knh = kn>>64, knl = kn;
      knh ^= 1ull<<63;
      int nzn;
      if(__builtin_expect(knh, 1)){
	nzn = __builtin_clzll(knh);
	knh = knh<<nzn|knl>>(-nzn&63);
	knl = knl<<nzn;
	kn = (u128)knh<<64|knl;
      } else {
	nzn = __builtin_clzll(knl);
	knl <<= nzn;
	kn = (u128)knl<<64;
	nzn += 64;
      }
      u64 tt = tn[isct];
      tt <<= 33 - e;
      X.b[1] += tt;
      int nzd = __builtin_clzll(X.b[1]);
      kd[2] = X.b[1]<<nzd|X.b[0]>>(-nzd&63);
      kd[1] = X.b[0]<<nzd;
      kd[0] = 0;
      dn = e + nzn - nzd + 15;
    }
  }
  
  u128 R = reciprocalU((u128)kd[2]<<64|kd[1]);
  u6x64 r = {0,0,0,0, R, R>>64};
  u5x64 H;
  mu5u2u3(H, r+4, kd);
  u3x64 sH;
  sH[0] = H[0]>>58|H[1]<<6;
  sH[1] = H[1]>>58|H[2]<<6;
  sH[2] = H[2]>>58|H[3]<<6;
  u6x64 H2;
  sqru6i3(H2,sH);
  shln(5, H, 11+64);
  u64 c;
  H[0] = __builtin_subcl(H[0], H2[2], 0, &c);
  H[1] = __builtin_subcl(H[1], H2[3], c, &c);
  H[2] = __builtin_subcl(H[2], H2[4], c, &c);
  H[3] = __builtin_subcl(H[3],     0, c, &c);
  H[4] = __builtin_subcl(H[4],     0, c, &c);
  u64 dR[7];
  mu7u5u2(dR, H, r+4);
  if(H[4]>>63){
    dR[5] = __builtin_subcl(dR[5], r[4], 0, &c);
    dR[6] = __builtin_subcl(dR[6], r[5], c, &c);
  }
  sarn(7,dR,10);
  r[0] = __builtin_subcl(r[0], dR[2], 0, &c);
  r[1] = __builtin_subcl(r[1], dR[3], c, &c);
  r[2] = __builtin_subcl(r[2], dR[4], c, &c);
  r[3] = __builtin_subcl(r[3], dR[5], c, &c);
  r[4] = __builtin_subcl(r[4], dR[6], c, &c);
  r[5] = __builtin_subcl(r[5], (i64)dR[6]>>63, c, &c);

  u64 t[8], nn[2] = {kn, kn>>64};
  mu8u6u2(t, r, nn);
  u6x64 t2;
  sqrhu6(t2, t+2);
  int dn2 = 2*dn-14;
  shrn(6, t2, dn2);

  static const u64 cp[] = {
    0x41df126e5ec4236b, 0x0000000000005029, 0xe7141aa59e9c06bb, 0x0000000005397794, 0xa29ae03dd40fb732, 0x000000572620ace8,
    0xf6c75ac2bf3ba522, 0x0005b05b05b058b0, 0x7f991c081742d872, 0x5f417d05f417cd73, 0xdfd4fc68937baa13, 0x7063e7063e706110,
    0x000000000000063e, 0xd2afeb0c4c44b156, 0x69069069069066d6, 0x0000000000690690, 0x3041b7df0f1e81af, 0x6eb3e45306eb3ce9,
    0x00000006eb3e4530, 0xf51a7ea3f72eba0a, 0x7507507507507456, 0x0000750750750750, 0x6d164965256c0f25, 0x7c1f07c1f07c1ebe,
    0x07c1f07c1f07c1f0, 0xed1197809680b661, 0x1084210842108407, 0x2108421084210842, 0x0000000000000084, 0x288d0c52f8e6282c,
    0x8d3dcb08d3dcb086, 0xb08d3dcb08d3dcb0, 0x000000000008d3dc, 0x81137e954e3d7cb3, 0xed097b425ed097b2, 0x097b425ed097b425,
    0x0000000097b425ed, 0x209a8d788f1f9fe1, 0xd70a3d70a3d70a3d, 0x3d70a3d70a3d70a3, 0x00000a3d70a3d70a, 0x09decb83cffc8425,
    0x590b21642c8590b2, 0x642c8590b21642c8, 0x00b21642c8590b21, 0x2f3708c2cde40e8c, 0xc30c30c30c30c30c, 0x0c30c30c30c30c30,
    0x30c30c30c30c30c3, 0x000000000000000c, 0x50b0362e3e44ed43, 0xd79435e50d79435e, 0x9435e50d79435e50, 0x35e50d79435e50d7,
    0x000000000000d794, 0x0f0bfeb7bb779a21, 0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f, 0x000000000f0f0f0f,
    0x1110e255d91317d2, 0x1111111111111111, 0x1111111111111111, 0x1111111111111111, 0x0000011111111111, 0xb13b119fb0d7521d,
    0x3b13b13b13b13b13, 0x13b13b13b13b13b1, 0xb13b13b13b13b13b, 0x0013b13b13b13b13, 0x1745d163a699798c, 0xd1745d1745d1745d,
    0x5d1745d1745d1745, 0x45d1745d1745d174, 0x745d1745d1745d17, 0x0000000000000001, 0x71c71c716c4c5694, 0xc71c71c71c71c71c,
    0x1c71c71c71c71c71, 0x71c71c71c71c71c7, 0xc71c71c71c71c71c, 0x0000000000001c71, 0x9249249247f4c207, 0x4924924924924924,
    0x2492492492492492, 0x9249249249249249, 0x4924924924924924, 0x0000000002492492, 0x33333333333116c0, 0x3333333333333333,
    0x3333333333333333, 0x3333333333333333, 0x3333333333333333, 0x0000003333333333, 0x55555555555553d6, 0x5555555555555555,
    0x5555555555555555, 0x5555555555555555, 0x5555555555555555, 0x0005555555555555};

  u6x64 f;
  const u64 *ck = cp;
  f[0] = ck[0];
  f[1] = ck[1];
  int i = 25;
  for(;i>20;--i){
    ck += 2;
    mhu2u2u2(f, f, t2+4);
    subu2u2u2(f, ck, f);
  }
  f[2] = ck[2];
  for(;i>15;--i){
    ck += 3;
    mhu3u3u3(f, f, t2+3);
    subu3u3u3(f, ck, f);
  }
  f[3] = ck[3];
  for(;i>10;--i){
    ck += 4;
    mhu4u4u4(f, f, t2+2);
    subu4u4u4(f, ck, f);
  }
  f[4] = ck[4];
  for(;i>5;--i){
    ck += 5;
    mhu5u5u5(f, f, t2+1);
    subu5u5u5(f, ck, f);
  }
  f[5] = ck[5];
  for(;i>1;--i){
    ck += 6;
    mhu6u6u6(f, f, t2);
    subu6u6u6(f, ck, f);
  }
  mhu6u6u6(f, f, t2);
  mhu6u6u6(f, t+2, f);
  subu6u6u6(f, t+2, f);

  if(isct|g){
    shrn(6,f,dn+1);
    addu6u6u6(f,phi0[isct],f);
    if(g) subu6u6u6(f, pio2, f);
    dn = -1;
  }
  b128u128_u v, dv;
  v.b[1] = f[5];
  v.b[0] = f[4];
  int k = __builtin_clzll(v.b[1]);
  v.a <<= k;
  xn = 0x3ffe - dn - k;
  u64 rnd = (v.b[0]>>14)&1;
  v.a >>= 15; // position mantissa
  if(__builtin_expect(rm != _MM_ROUND_NEAREST, 0))
    rnd = !xsgn*(rm==_MM_ROUND_UP) + !!xsgn*(rm==_MM_ROUND_DOWN);
  dv.b[0] = rnd;
  dv.b[1] = (u64)xn<<48;
  v.a += dv.a;
  v.b[1] |= xsgn;
  flagp |= FE_INEXACT;
  if(__builtin_expect(oflagp!=flagp, 0)) _mm_setcsr(flagp);
  __float128 res = reinterpret_u128_as_f128(v.a); // put into xmm register
  return res;
}

#ifndef __APPLE__
// somewhat we need to include that for icx and the Intel math library
extern __float128 __atanq (__float128, __float128);

// atanq is called atanf128 in GNU libc, and __atanq in the Intel math library
__float128 atanq(__float128 x) {
#ifdef __INTEL_CLANG_COMPILER
  return __atanq (x);
#else
  return atanf128 (x);
#endif
}
#endif
