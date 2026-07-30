# The CORE-MATH project

CORE-MATH Mission: provide on-the-shelf open-source mathematical
functions with correct rounding that can be integrated into current
mathematical libraries (GNU libc, Intel Math Library, AMD Libm,
Newlib, OpenLibm, Musl, Apple Libm, llvm-libc, CUDA libm, ROCm).

Homepage: https://core-math.gitlabpages.inria.fr/


## Quick guide

### Exhaustive checks

To run an exhaustive check of a single-precision single-argument
function, run:

    ./check.sh --exhaustive [rounding_modes] $FUN

where:
- `$FUN` can be `acosf`, `asinf`, etc.
- `[rounding_modes]` can be a selection of `--rndn` (round to
  nearest), `--rndz` (toward zero), `--rndu` (upwards), `--rndd`
  (downwards). The default is all four.

This command is sensitive to the following environment variables:
- `CC`
- `CFLAGS`
- OpenMP variables such as `OMP_NUM_THREADS`

Note: on Debian, you need the libomp-dev package to use clang.

### Worst case checks

These checks are available for bivariate single-precision functions,
and double-precision functions.

    ./check.sh --worst [rounding_modes] $FUN

### Special checks

These checks are available for functions where some interesting worst
cases can be automatically (and cheaply) computed (at the time of
writing, only `hypotf` and `cbrt`).

    ./check.sh --special [rounding_modes] $FUN

### Performance measurements

Performance measurement scripts rely on the Linux perf framework. You
might need to run the command:

    echo -1 > /proc/sys/kernel/perf_event_paranoid

as root before running the following commands.

To evaluate the reciprocal throughput of some function, run:

    ./perf.sh acosf

It outputs two numbers: the performance of core-math, and the one the
libc, in cycles/call. You can also evaluate the performance of some
other libm (given by a static .a archive, typically llvmlibc), by
setting the `LIBM` environment variable to the absolute path of the .a
file. In the case, `./perf.sh` will output three numbers: the
performances of core-math, standard libm, given libm.

To evaluate performance of all supported functions, run:

    ./perf-all.sh

You can also set the `PERF_ARGS` environment variable to `--latency`
to get latency instead of reciprocal throughput.

When you run ./perf.sh acosf, it does the following:

   $ export OPENMP=-fopenmp
   $ cd src/binary32/acos
   $ make clean
   $ make CFLAGS="-O3 -march=native"
   $ ./perf --file /tmp/randoms.dat --reference --count 1000000
   $ perf stat ./perf --file /tmp/randoms.dat --count 1000000 --repeat 1000

and it reports the number of cycles given by perf (divided by 10^9).

## Layout

Each function `$NAME` has a dedicated directory
`src/$TYPE/$SHORT_NAME`, where `$TYPE` can be `binary{32,64,80,128}`
and `$SHORT_NAME` is the function name without its type suffix
(`acos`, `asin`, etc.). This directory contains the following files:
- `$NAME.c`: a standalone implementation of function `cr_$NAME`
- other support files


## How to add support for a new function?

You can take inspiration from a similar function of the same bitsize
and/or arity.

For example, suppose you want to add support for a binary64 bivariate
function `foo`. In `src/binary64/foo`, you will need the following
files:
- `foo.c`: defining `cr_foo`
- `foo.wc`: the worst cases to be tested
- `foo_mpfr.c`: defining `ref_foo`, using MPFR
- `Makefile`: defining `FUNCTION_UNDER_TEST`, and including `../support/Makefile.bivariate`
- `function_under_test.h`: defining `{cr,ref}_function_under_test`

In addition, in `src/generic/foo`, you will need the following file:
- `random_under_test.h`: defining a function `random_under_test`,
  which samples suitable inputs for `foo`

## Support of underflow

CORE-MATH always supports underflow, with the rule of "underflow after
rounding": if the rounded result (with unbounded exponent range and full
target precision) is in the subnormal range, then underflow is raised.
Otherwise underflow is not raised. As in the C standard, underflow is not
raised when the result is exact. CORE-MATH assumes that the processor also
supports underflow after rounding, since in some cases raising underflow
is delegated to the underlying processor operations, as in "return a+b".

If you use the CORE-MATH test suite (check.sh) to check another library,
you can control how underflow is checked with CORE_MATH_NOCHECK_UNDERFLOW:

CORE_MATH_NOCHECK_UNDERFLOW=0 assumes underflow after rounding
CORE_MATH_NOCHECK_UNDERFLOW=1 assumes underflow before rounding
CORE_MATH_NOCHECK_UNDERFLOW=-1 means that underflow is not checked when
   underflow before rounding and underflow after rounding disagree.

If the external library supports underflow before rounding, add
-DCORE_MATH_UNDERFLOW_BEFORE to EXTRA_CFLAGS in check.sh.

## Support of overflow

CORE-MATH always supports overflow: if the rounded result (with unbounded
exponent range) is not representable, then overflow is raised.
Otherwise overflow is not raised.

## Support of inexact

The support of the inexact exception is provided with -DCORE_MATH_CHECK_INEXACT
(not activated by default). When CORE_MATH_CHECK_INEXACT is defined, the
inexact exception (from fenv.h) is raised when the result is inexact, and
is not raised when the result is exact.

## Support of errno

The support of errno is provided with -DCORE_MATH_SUPPORT_ERRNO
(not activated by default). The value of errno is set to ERANGE in
case of overflow or of underflow (following the "underflow after
rounding" rule). The value of errno is set to EDOM in case of domain
error. When the input is NaN or Inf, errno is not changed.

Several CORE-MATH functions use __builtin_fma or __builtin_fmaf,
which may call the corresponding fma or fmaf library functions if not
available in hardware. CORE-MATH assumes these library functions
do not set spurious errno.

## Notes

The CORE-MATH code assumes all computations are rounded at runtime with the
given rounding mode, for example 1.0f - 0x1p-25f should round to 1.0f in
binary32 for rounding to nearest or upward, and to nextdown(1) for rounding
to zero or downward. This requires the option -frounding-math on some compilers.

The CORE-MATH code assumes all double-precision computations are rounded to
double precision. On x86 processors where these computations are performed
on the x87 FPU, the user should set up the rounding precision to double
precision (on Linux it is set to double-extended by default).

There might be some contradictions between correct rounding and some optional
requirements of the POSIX standard. In such a case, CORE-MATH chooses to
return the correctly rounded result.

## Known issues

The Apple ldexp() function has a bug in the subnormal range (Darwin 25.1.0).
For example ldexp(0x1.9192d74f53405p-6, -1027) gives 0x1.9192d74f53p-1033
instead of 0x1.9192d74f538p-1033. This was reported at FB21774989 and FB21774410.
This yields a failure in CORE-MATH erfc.

For efficiency, some CORE-MATH routines directly play with the fenv.h macros
(FE_TONEAREST, FE_UPWARD, FE_DOWNWARD). However, these macros might have
different values under different operating systems, or even between different
versions of the same operating system (for example, Windows 10 version 14393
swapped the values of FE_UPWARD and FE_DOWNWARD). This might yield failures.

CORE-MATH assumes all floating-point operations are done with the target
precision (FLT_EVAL_METHOD=0 in the C standard).
On x86 32-bit platforms, you might have FLT_EVAL_METHOD=2 by default,
with computations done in double-extended precision using the x87 coprocessor.
To disable this, you might have to add -fexcess-precision=standard
(for gcc) to ensure the compiler performs internal computations in double
precision (and not in extended double precision).

Another issue with 32-bit platforms is that some of them miss support
for __uint128_t or _BitInt(128). The functions using these types are
thus not available.
