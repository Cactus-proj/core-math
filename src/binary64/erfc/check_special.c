/* Check erfc on special and random inputs.

Copyright (c) 2022-2024 Paul Zimmermann, Inria.

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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fenv.h>
#include <math.h>
#include <sys/types.h>
#include <unistd.h>
#include <mpfr.h>
#if (defined(_OPENMP) && !defined(CORE_MATH_NO_OPENMP))
#include <omp.h>
#endif
#include "function_under_test.h"

#ifndef CORE_MATH_TESTS
#define CORE_MATH_TESTS 200000000UL /* total number of tests */
#endif

int ref_init (void);
int ref_fesetround (int);

double cr_erfc (double);
double ref_erfc (double);

int rnd1[] = { FE_TONEAREST, FE_TOWARDZERO, FE_UPWARD, FE_DOWNWARD };

int rnd = 0;
int verbose = 0;
unsigned long tested = 0;

static inline double tfun(double x){
  return cr_function_under_test(x);
}

#define MAX_THREADS 192

static unsigned int Seed[MAX_THREADS];

static inline uint64_t
asuint64 (double f)
{
  union
  {
    double f;
    uint64_t i;
  } u = {f};
  return u.i;
}

static inline double
asfloat64 (uint64_t i)
{
  union
  {
    uint64_t i;
    double f;
  } u = {i};
  return u.f;
}

typedef union {double f; uint64_t u;} b64u64_u;

static double
get_random (int tid)
{
  b64u64_u v;
  v.u = rand_r (Seed + tid);
  v.u |= (uint64_t) rand_r (Seed + tid) << 31;
  v.u |= (uint64_t) rand_r (Seed + tid) << 62;
  return v.f;
}

static void
check (double x)
{
#pragma omp atomic update
  tested ++;
  int bug;
  double y1 = ref_erfc (x);
  fesetround (rnd1[rnd]);
  double y2 = cr_erfc (x);
  if (isnan (y1))
    bug = !isnan (y2);
  else if (isnan (y2))
    bug = !isnan (y1);
  else
    bug = asuint64 (y1) != asuint64 (y2);
  if (bug)
  {
    printf ("FAIL x=%la ref=%la z=%la\n", x, y1, y2);
    fflush (stdout);
#ifndef DO_NOT_ABORT
    exit (1);
#endif
  }
}

/* check when ulp(erfc(x)) lies in the subnormal range:
   0x1.9db1bb14e15cap+4 <= x <= 0x1.b39dc41e48bfcp+4 */
static void
check_subnormal (void)
{
  double xmin = 0x1.9db1bb14e15cap+4;
  double xmax = 0x1.b39dc41e48bfcp+4;
  uint64_t umin = asuint64 (xmin);
  uint64_t umax = asuint64 (xmax);
  uint64_t urange = (umax - umin) / (uint64_t) CORE_MATH_TESTS;
  /* we multiply urange by 100 since tests in the subnormal range are more
     expensive */
  urange ++; // +1 to avoid urange=0
  printf ("Check subnormal output range\n");
  umin += getpid () % urange;
#if (defined(_OPENMP) && !defined(CORE_MATH_NO_OPENMP))
#pragma omp parallel for
#endif
  for (uint64_t u = umin; u <= umax; u += urange)
  {
    ref_init ();
    ref_fesetround (rnd);
    check (asfloat64 (u));
  }
}

// put in h+l a double-double approximation of erfc(x)
static void
dd_erfc (double *h, double *l, double x)
{
  mpfr_t t;
  mpfr_init2 (t, 107);
  mpfr_set_d (t, x, MPFR_RNDN);
  mpfr_erfc (t, t, MPFR_RNDN);
  *h = mpfr_get_d (t, MPFR_RNDN);
  mpfr_sub_d (t, t, *h, MPFR_RNDN);
  *l = mpfr_get_d (t, MPFR_RNDN);
  mpfr_clear (t);
}

// return a double approximation of the derivative of erfc(x)
static double
deriv (double x)
{
  mpfr_t t;
  // the derivative of erfc(x) is -2*e^(-x^2)/sqrt(pi)
  mpfr_init2 (t, 53);
  mpfr_set_d (t, x, MPFR_RNDN);
  mpfr_sqr (t, t, MPFR_RNDN); // t = x^2
  mpfr_neg (t, t, MPFR_RNDN); // t = -x^2
  mpfr_exp (t, t, MPFR_RNDN); // t = exp(-x^2)
  double ret = mpfr_get_d (t, MPFR_RNDN);
  mpfr_clear (t);
  return -0x1.20dd750429b6dp+0 * ret;
}

static void scan_consecutive_aux(int64_t n, double x){
  ref_init();
  ref_fesetround(rnd);
  fesetround(rnd1[rnd]);
  while (n) {
    double h, l, d, dd;
    dd_erfc (&h, &l, x);
    int e;
    frexp (x, &e);
    d = deriv (x); // derivative of erfc(x)
    dd = fabs (-2.0 * x * d); // absolute value of 2nd derivative
    /* 2^(e-1) <= |x| < 2^e thus ulp(x) = 2^(e-53) */
    d = ldexp (d, e - 53); // multiply d by ulp(x)
    dd = ldexp (dd, 2 * (e - 53)); // multiply dd by ulp(x)^2
    /* we want j^2*dd < 2^-11 ulp(h) so that the 2nd-order term
       produces an error bounded by 2^-11 ulp(h), to that MPFR
       will be called with probability about 2^-11.
       Thus approximately j^2*dd < 2^-64 h,
       or j < 2^-32 sqrt(h/dd) */
    int64_t jmax = 0x1p-32 * sqrt (h / dd);
    if (jmax > n) jmax = n; // cap to n
    if (jmax == 0) jmax = 1; // ensure progress
    for(int64_t j=0;j<jmax;j++){
      b64u64_u v = {.f = x};
      v.u += j;
      double t = tfun (v.f);
      // erfc(x+j*u) is approximated by h + l + j*d
      double w = h + __builtin_fma (j, d, l);
      if (t != w) // expensive test
        check(v.f);
    }
    n -= jmax;
    x += jmax * ldexp (1.0, e - 53);
  }
}

static void scan_consecutive (int64_t n, double x){
  int nthreads = 1;
  if (n < 0) {
    n = -n;
    x = asfloat64 (asuint64 (x) - n);
  }
#if (defined(_OPENMP) && !defined(CORE_MATH_NO_OPENMP))
#pragma omp parallel
  nthreads = omp_get_num_threads ();
#endif
  int64_t h = (n - 1) / nthreads + 1; // ceil(n/nthreads)
#if (defined(_OPENMP) && !defined(CORE_MATH_NO_OPENMP))
#pragma omp parallel for schedule(static,1)
#endif
  for (int i = 0; i < nthreads; i++) {
    int64_t ni = i * h;
    double xi = asfloat64 (asuint64 (x) + ni);
    int64_t hi = (ni + h > n) ? n - ni : h;
    scan_consecutive_aux (hi, xi);
  }
  printf ("checked %lu values, expensive checks %lu\n",
          (unsigned long) n, tested);
}

int
main (int argc, char *argv[])
{
  int conseq = 0;
  unsigned long n = 1000000; // length of consecutive runs with -C
  double a = 1.5;  // starting point of consecutive runs with -C

  while (argc >= 2)
    {
      if (strcmp (argv[1], "--rndn") == 0)
        {
          rnd = 0;
          argc --;
          argv ++;
        }
      else if (strcmp (argv[1], "--rndz") == 0)
        {
          rnd = 1;
          argc --;
          argv ++;
        }
      else if (strcmp (argv[1], "--rndu") == 0)
        {
          rnd = 2;
          argc --;
          argv ++;
        }
      else if (strcmp (argv[1], "--rndd") == 0)
        {
          rnd = 3;
          argc --;
          argv ++;
        }
      else if (strcmp (argv[1], "-C") == 0)
        {
          conseq = 1;
          n = strtoul (argv[2], NULL, 0);
          argc -= 2;
          argv += 2;
        }
      else if (strcmp (argv[1], "-a") == 0)
        {
          a = strtod (argv[2], NULL);
          argc -= 2;
          argv += 2;
        }
      else if (strcmp (argv[1], "--verbose") == 0)
        {
          verbose = 1;
          argc --;
          argv ++;
        }
      else
        {
          fprintf (stderr, "Error, unknown option %s\n", argv[1]);
          exit (1);
        }
    }

  if (conseq) {
    scan_consecutive (n, a);
    return 0;
  }

  check_subnormal ();

  printf ("Random tests\n");
  unsigned int seed = getpid ();
  for (int i = 0; i < MAX_THREADS; i++)
    Seed[i] = seed + i;

#define XMAX 0x1.b39dc41e48bfdp+4
#define XMIN -0x1.7744f8f74e94bp2
  
#if (defined(_OPENMP) && !defined(CORE_MATH_NO_OPENMP))
#pragma omp parallel for
#endif
  for (n = 0; n < CORE_MATH_TESTS; n++)
  {
    ref_init ();
    ref_fesetround (rnd);
    double x;
    int tid;
#if (defined(_OPENMP) && !defined(CORE_MATH_NO_OPENMP))
    tid = omp_get_thread_num ();
#else
    tid = 0;
#endif
    do x = get_random (tid); while (x < XMIN || XMAX < x);
    check (x);
  }

  return 0;
}
