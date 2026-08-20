/* Correctly-rounded mpfr-based arc tangent in binary128 float point format

Copyright (c) 2026 Alexei Sibidanov <sibid@uvic.ca>.

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

#include <mpfr.h>
#include "fenv_mpfr.h"
#include <stdint.h>

typedef unsigned __int128 u128;
typedef uint64_t u64;
typedef union {u128 a; u64 b[2]; __float128 f;} b128u128_u;

__float128 ref_atanq(__float128 x){
  /* since MPFR does not distinguish between quiet/signaling NaN,
     we have to deal with them separately to apply the IEEE rules */
  b128u128_u xi = {.f = x};
  if((xi.a<<1)<((u128)0xffffull<<112) && (xi.a<<1)>((u128)0x7fffull<<113)){ // x = sNAN
    xi.b[1] |= 1ull<<47;
    return xi.f; // return qNAN
  }
  mpfr_t y;
  mpfr_init2 (y, 113);
  mpfr_set_float128 (y, x, MPFR_RNDN);
  int inex = mpfr_atan (y, y, rnd2[rnd]);
  mpfr_subnormalize (y, inex, rnd2[rnd]);
  __float128 ret = mpfr_get_float128 (y, MPFR_RNDN);
  mpfr_clear (y);
  return ret;
}
