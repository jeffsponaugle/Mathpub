# psph -- exhaustive search for extremal postage stamp bases
# macOS/Apple Silicon: make            (clang, NEON)
# Linux x86-64:        make linux      (gcc or clang, AVX2 via -march=native; add NOSIMD=1 for scalar)
# Linux aarch64:       make linux      (NEON)
#
# Targets: psph (8-bit stamp counts, h <= 254), psph16 (16-bit counts, h <= 65534), linux (both, with gcc and
# -march=native), selftest (quick selftest, ~4 min on 10 threads), clean.  The CUDA build lives in ../psph_cuda.
CC      ?= clang
ARCHFLAGS ?= -mcpu=apple-m1
CFLAGS  = -O3 $(ARCHFLAGS) -std=c11 -pthread -Wall -Wextra -Wno-unused-parameter $(if $(NOSIMD),-DNO_SIMD,)
LDFLAGS = -lm

all: psph psph16

psph: psph.c
	$(CC) $(CFLAGS) -o psph psph.c $(LDFLAGS)

# 16-bit stamp counts: needed for h >= 255 (e.g. k=4, h=302/303); ~1.6x slower per candidate
psph16: psph.c
	$(CC) $(CFLAGS) -DWIDE -o psph16 psph.c $(LDFLAGS)

# rebuild both binaries with gcc (LCC=clang to override) for the CPU of the machine it runs on
linux:
	$(MAKE) CC=$(or $(LCC),gcc) ARCHFLAGS=-march=native all

selftest: psph
	./psph -selftest quick

clean:
	rm -f psph psph16

.PHONY: all linux selftest clean
