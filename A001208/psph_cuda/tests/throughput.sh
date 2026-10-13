#!/bin/sh
# Throughput + full-check-flavour cross-checks for the GPU leaf (run on orin1 from ~/A001208/psph_cuda after validate.sh).
# Logs in logs/throughput.log.  The CPU on orin1 is shared with the production run, so CPU-side numbers are pessimistic;
# the "[gpu] GPU time" lines (CUDA events) are not affected by CPU contention.
#
# Usage (on orin1): sh tests/throughput.sh   (J = worker threads, default 6; uses GNU timeout, 900 s per run).
# Measures CPU vs GPU leaf throughput on (9,8) and (27,6) item ranges, and cross-checks the two full-check kernel
# flavours (PSPH_GPU_FULL=thread / warp, plus warp with PARANOID) against the CPU on k=4 h=60 and k=5 h=25, where
# a_k is large and full checks are frequent: the sorted solution lists must be identical.
cd ~/A001208/psph_cuda || exit 1
mkdir -p logs
N="nice -n 10"
J=${J:-6}
L=logs/throughput.log
echo "=== throughput.sh start $(date) J=$J; load: $(cat /proc/loadavg)" > $L
# run TAG BINARY ARGS...: one timed run (nice 10, at most 900 s) with its key statistics appended to $L.
run() {  # tag binary args...   (env may be set by the caller)
  tag=$1; bin=$2; shift 2
  echo "--- $tag: $bin $*   start $(date)" >> $L
  t0=$(date +%s); $N timeout 900 ./$bin "$@" -j $J > logs/$tag.out 2> logs/$tag.err; rc=$?; t1=$(date +%s)
  echo "    rc=$rc wall $((t1-t0)) s  solutions: $(grep -c SOLUTION logs/$tag.out)" >> $L
  grep -h 'leaf candidates\|probes:\|^\[psph\] wall\|GPU time\|batches\|^wall\|FATAL\|PARANOID\|INTERNAL' logs/$tag.err | sed 's/^/    /' >> $L
}
# (9,8) TGT 5600 items 10900:11003: CPU reference and GPU
run tp_h9k8_cpu psph_cpu -h 9 -k 8 -t 5600 -d 4 -i 10900:11003 -p 0
run tp_h9k8_gpu psph_gpu -h 9 -k 8 -t 5600 -d 4 -i 10900:11003 -p 0
PSPH_GPU_STRATA=1 run tp_h9k8_gpu_s1 psph_gpu -h 9 -k 8 -t 5600 -d 4 -i 10900:11003 -p 0
# (27,6) TGT 176381, depth 3, items 0:20
run tp_h27k6_gpu psph_gpu -h 27 -k 6 -t 176381 -d 3 -i 0:20 -p 60
run tp_h27k6_cpu psph_cpu -h 27 -k 6 -t 176381 -d 3 -i 0:20 -p 60
# full-check flavours must agree with each other and with the CPU (k=4 h=60: large a_4, many full checks)
run fc_h4k60_cpu    psph_cpu -h 60 -k 4 -t 143814 -p 0
PSPH_GPU_FULL=thread run fc_h4k60_thread psph_gpu -h 60 -k 4 -t 143814 -p 0
PSPH_GPU_FULL=warp   run fc_h4k60_warp   psph_gpu -h 60 -k 4 -t 143814 -p 0
PSPH_GPU_FULL=warp PSPH_GPU_PARANOID=1 run fc_h4k60_warp_par psph_gpu -h 60 -k 4 -t 143814 -p 0
run fc_h5k25_cpu    psph_cpu -h 25 -k 5 -t 31108 -p 0
PSPH_GPU_FULL=thread run fc_h5k25_thread psph_gpu -h 25 -k 5 -t 31108 -p 0
PSPH_GPU_FULL=warp   run fc_h5k25_warp   psph_gpu -h 25 -k 5 -t 31108 -p 0
for t in fc_h4k60 fc_h5k25; do
  sort logs/${t}_cpu.out > logs/${t}_cpu.sorted
  for f in thread warp; do sort logs/${t}_$f.out > logs/${t}_$f.sorted; cmp -s logs/${t}_cpu.sorted logs/${t}_$f.sorted && echo "    $t $f: SOLUTIONS IDENTICAL to CPU" >> $L || echo "    $t $f: *** DIFFER ***" >> $L; done
  echo "    $t X-pass cpu/thread/warp: $(grep -o 'X-pass: [0-9]*' logs/${t}_cpu.err) / $(grep -o 'X-pass: [0-9]*' logs/${t}_thread.err) / $(grep -o 'X-pass: [0-9]*' logs/${t}_warp.err)" >> $L
  echo "    $t full-check fail thread/warp: $(grep -o 'full-check fail: [0-9]*' logs/${t}_thread.err) / $(grep -o 'full-check fail: [0-9]*' logs/${t}_warp.err)" >> $L
done
echo "=== throughput.sh done $(date)" >> $L
