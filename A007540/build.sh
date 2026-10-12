#!/bin/sh
# Build Gerbicz/Costa/Harvey Wilson-prime search (pw13) + Harvey's ntt library.
# Needs: a C compiler with OpenMP (Homebrew gcc on macOS: brew install gcc; gcc on Linux),
#        GMP *source* tree (pw13 includes gmp-impl.h/longlong.h, internal headers).
# Usage: ./build.sh [THREADS]   -> builds ./pw13, ./wilsontest, ./tools/wilson_brute
#        THREADS (default: nproc) is the max thread count for the ntt tuner.
set -e
cd "$(dirname "$0")"
THREADS=${1:-$(getconf _NPROCESSORS_ONLN)}
case "$(uname)" in
  Darwin) CC=${CC:-$(ls /opt/homebrew/bin/gcc-[0-9]* | sort -V | tail -1)};;
  *)      CC=${CC:-gcc};;
esac
GMPV=6.3.0
if [ ! -f third_party/gmp-$GMPV/.libs/libgmp.a ]; then
  cd third_party
  [ -f gmp-$GMPV.tar.xz ] || curl -sLO https://gmplib.org/download/gmp/gmp-$GMPV.tar.xz
  tar xJf gmp-$GMPV.tar.xz && cd gmp-$GMPV && ./configure --disable-shared >/dev/null && make -j"$THREADS" >/dev/null
  cd ../..
fi
GMP=$PWD/third_party/gmp-$GMPV
cd third_party/Wilson-prime
ln -sfn ../gmp-$GMPV gmp-5.0.4          # pw13.c hardcodes this include path
ln -sfn ntt/ntt-0.1.2 ntt-0.1.2
NTT="ntt-0.1.2/profile.c ntt-0.1.2/misc.c ntt-0.1.2/modarith.c ntt-0.1.2/memory.c ntt-0.1.2/fft_main.c ntt-0.1.2/fft_base.c ntt-0.1.2/fft_array.c ntt-0.1.2/intmult.c ntt-0.1.2/tunetab.c"
FLAGS="-fopenmp -fgnu89-inline -std=gnu99 -O2 -w -I. -I$GMP"
# 1) tune the ntt library for this machine (shipped tunetab.c DISABLES ntt: tune_tab[1]=SIZE_MAX)
if ! grep -q "tuned-for" ntt-0.1.2/tunetab.c; then
  (cd ntt-0.1.2 && $CC $FLAGS -o tune tune.c profile.c misc.c modarith.c memory.c fft_main.c fft_base.c fft_array.c intmult.c tunetab.c $GMP/.libs/libgmp.a -lm \
     && echo "running ntt tuner for $THREADS threads (several minutes)..." && ./tune "$THREADS" > tunetab.new 2> tune.log \
     && { echo "// tuned-for $(hostname) $(date +%F)"; cat tunetab.new; } > tunetab.c)
fi
$CC $FLAGS -o ../../pw13 pw13.c $NTT $GMP/.libs/libgmp.a -lm
$CC -O2 -w -o ../../wilsontest wilsontest.c -I$GMP $GMP/.libs/libgmp.a -lm
cd ../..
cc -O2 -o tools/wilson_brute tools/wilson_brute.c -lpthread
cc -O2 -o tools/wilsonnum tools/wilsonnum.c -lpthread -lm          # A157250 Wilson-number sieve (see A157250.md)
cc -O2 -o tools/wilsonnum_brute tools/wilsonnum_brute.c -lpthread
echo "built: pw13 wilsontest tools/wilson_brute tools/wilsonnum tools/wilsonnum_brute"
