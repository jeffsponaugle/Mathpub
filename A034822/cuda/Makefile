# CUDA version of the A034822 search (DGX Spark / GB10: sm_121).
NVCC    ?= /usr/local/cuda/bin/nvcc
ARCH    ?= sm_121
NVFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -DMINB=2 -Xcompiler -pthread -Xcompiler -Wall -Xptxas -v

a034822_cuda: a034822_cuda.cu
	$(NVCC) $(NVFLAGS) -o $@ $< 2> build.log || (cat build.log; false)
	@grep -E "registers|spill" build.log | sort | uniq -c | head -40

clean:
	rm -f a034822_cuda build.log
.PHONY: clean
