/* Correctly-rounded Euclidean distance function (hypot) for bfloat16 value.

Copyright (c) 2025-2026 Maxence Ponsardin and Paul Zimmermann

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

#include <stdint.h>
#ifdef CORE_MATH_SUPPORT_ERRNO
#include <errno.h>
#endif
#include <math.h> // only used during performance tests

// Warning: clang also defines __GNUC__
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#endif

#pragma STDC FENV_ACCESS ON

typedef union {double f; uint64_t u;} b64u64_u;
typedef union {__bf16 f; uint16_t u;} b16u16_u;

static inline uint16_t
asuint (__bf16 f)
{
  b16u16_u u = {.f = f};
  return u.u;
}

static inline int
is_inf (uint16_t u)
{
  // +Inf is encoded as 0x7f80, and -Inf as 0xff80
  return u == 0x7f80 || u == 0xff80;
}

static inline int
is_qnan (uint16_t u)
{
  // +qnan is encoded as 0x7fc0 to 0x7fff
  return 0x7fc0 <= (u & 0x7fff);
}

__bf16 cr_hypot_bf16(__bf16 x, __bf16 y){
  b16u16_u sx = {.f = x}, sy = {.f = y};
  b64u64_u tx = {.f = x}, ty = {.f = y};
  /* we cast the inputs to double precision, since float is not enough,
     for example with x,y=nan,0x1p-133 we would get a spurious inexact
     exception */
  double z = __builtin_sqrt (tx.f * tx.f + ty.f * ty.f);

  // hypot(+/-inf,qnan) = +inf
  if (is_inf (sx.u) && is_qnan (sy.u)) return tx.f * tx.f;
  if (is_inf (sy.u) && is_qnan (sx.u)) return ty.f * ty.f;

  __bf16 ret = z;
#ifdef CORE_MATH_SUPPORT_ERRNO
  /* there is underflow for z < 0x1p-126 for rndz/rndd,
     for z < 0x1.ffp-127 for rndn and
     for z <= 0x1.fep-127 for rndu
     (there is no underflow if z is exact) */
  if (z != 0 && (double) ret != z && z < 0x1p-126) {
    int t = 1.0f + 0x1p-24f > 1.0f; // true for rndu only
    int u = 1.0f - 0x1p-25f < 1.0f; // true for rndz/rndd only
    if (u || (z < 0x1.ffp-127 && !t) || (z <= 0x1.fep-127 && t))
      errno = ERANGE; // underflow
  }
  /* there is overflow for z >= 0x1p+128 for rndu,
     for z >= 0x1.ffp+127 for rndn and
     for z > 0x1.fep+127 for rndz/rndd.
     We limit z to 2^129, otherwise for x,y=inf,0x0p+0 we get a spurious
     inexact due to the computation of t and u. */
  if (z > 0x1.fep+127 && z < 0x1p129) {
    int t = 1.0f + 0x1p-24f > 1.0f; // true for rndu only
    int u = 1.0f - 0x1p-25f < 1.0f; // true for rndz/rndd only
    if (t || (z >= 0x1.ffp+127 && !u) || (z >= 0x1p+128 && u))
      errno = ERANGE; // overflow
  }
#endif
  return ret;
}

// dummy function since GNU libc does not provide it
__bf16 hypot_bf16 (__bf16 x, __bf16 y) {
	return (__bf16) hypotf ((float) x, (float) y);
}
