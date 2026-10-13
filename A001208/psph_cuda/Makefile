# psph_cuda -- CUDA leaf for psph (postage stamp exhaustive search)
#
#   make                 psph_gpu (8-bit cells, h <= 254) and psph_gpu16 (16-bit cells, -DWIDE, h <= 65534)
#   make psph_cpu        reference CPU binary from the same psph.c (for differential tests)
#
# Jetson Orin (sm_87) is the default.  DGX Spark (GB10, Blackwell): make ARCH="-arch=sm_121"  (or -arch=native
# with CUDA >= 12.8).  Any other GPU: ARCH="-arch=native".  The host part is compiled with gcc -march=native
# (NEON on aarch64, AVX2 on x86) exactly like the CPU program.
#
# psph.c is taken from ../psph/psph.c unless a copy psph.c exists in this directory (deployment layout).
#
# The host part (psph.c) is compiled with gcc and linked by nvcc together with gpu_leaf.cu; both must be built with
# the same -DWIDE setting because the table cell type is shared.  "make selftest" runs the quick selftest in GPU mode.
# Validation and throughput scripts: tests/validate.sh, tests/throughput.sh.
NVCC     ?= /usr/local/cuda/bin/nvcc
ARCH     ?= -arch=sm_87
CC       ?= gcc
PSPH_SRC ?= $(if $(wildcard psph.c),psph.c,../psph/psph.c)
CFLAGS    = -O3 -march=native -std=gnu11 -pthread -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -I. -DGPU_LEAF
NVFLAGS   = $(ARCH) -O3 -std=c++17 -lineinfo -Xcompiler -O3,-Wall,-Wno-unused-function
LDLIBS    = -lpthread -lm

all: psph_gpu psph_gpu16

psph_gpu: psph_gpu.o gpu_leaf.o
	$(NVCC) $(ARCH) -o $@ $^ $(LDLIBS)

psph_gpu.o: $(PSPH_SRC) gpu_leaf.h
	$(CC) $(CFLAGS) -c -o $@ $(PSPH_SRC)

gpu_leaf.o: gpu_leaf.cu gpu_leaf.h
	$(NVCC) $(NVFLAGS) -c -o $@ gpu_leaf.cu

psph_gpu16: psph_gpu16.o gpu_leaf16.o
	$(NVCC) $(ARCH) -o $@ $^ $(LDLIBS)

psph_gpu16.o: $(PSPH_SRC) gpu_leaf.h
	$(CC) $(CFLAGS) -DWIDE -c -o $@ $(PSPH_SRC)

gpu_leaf16.o: gpu_leaf.cu gpu_leaf.h
	$(NVCC) $(NVFLAGS) -DWIDE -c -o $@ gpu_leaf.cu

# reference CPU binaries (no GPU code) from the same source
psph_cpu: $(PSPH_SRC)
	$(CC) -O3 -march=native -std=gnu11 -pthread -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -o $@ $(PSPH_SRC) -lm

psph_cpu16: $(PSPH_SRC)
	$(CC) -O3 -march=native -std=gnu11 -pthread -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -DWIDE -o $@ $(PSPH_SRC) -lm

selftest: psph_gpu
	./psph_gpu -selftest quick -j 6

clean:
	rm -f psph_gpu psph_gpu16 psph_cpu psph_cpu16 *.o

.PHONY: all selftest clean
