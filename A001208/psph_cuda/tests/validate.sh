#!/bin/sh
# Validation suite for the GPU leaf, run on orin1 from ~/A001208/psph_cuda.  Logs in logs/.
# 1. selftest (GPU mode)   2. differential tests GPU vs CPU binary (same source)   3. -s2 0 exact-match tests
#
# Usage (on orin1):  cd ~/A001208/psph_cuda && make all psph_cpu psph_cpu16 && LOG=logs/validate_diff.log sh tests/validate.sh
#   LOG must be given by the caller: the line LOG=${LOG:-$LOG} below supplies no default (with LOG unset every
#   "> $LOG" redirection fails).  J = worker threads (default 6); SKIP_SELFTEST=1 skips step 1.  Uses GNU timeout.
# Method: psph_cpu and psph_gpu are the SAME psph.c built without and with the CUDA leaf.  For each case both run the
#   same search; the sorted SOLUTION lists must be identical, and so must the X-pass counts (stage 1 is replicated
#   exactly on the GPU).  The GPU picks its deep holes per prefix instead of per candidate, so the later counters may
#   differ; with -s2 0 (no deep holes) every counter must agree.  par_case additionally forces the 64-bit filter
#   kernel and the PARANOID host re-check of every batch.
# Output: one block per case in $LOG ("SOLUTIONS IDENTICAL ..." / "*** SOLUTION SETS DIFFER ***", X-pass verdict,
#   leaf statistics, wall times, GPU time, any PARANOID/FATAL/INTERNAL lines); per-case .out/.err/.sorted in logs/.
#   Result of the Oct 1 2026 run: every case that finished gave identical solution sets; two (9,8) cases hit the
#   900 s limit on both binaries and were replaced by smaller item ranges (see psph_cuda/README.md, Validation).
cd ~/A001208/psph_cuda || exit 1
mkdir -p logs
N="nice -n 10"
J=${J:-6}
LOG=${LOG:-$LOG}
echo "=== validate.sh start $(date)  J=$J  load $(cat /proc/loadavg)" > $LOG
# ---- 1. selftest (SKIP_SELFTEST=1 to skip)
if [ -z "$SKIP_SELFTEST" ]; then
  echo "=== selftest quick (GPU) start $(date)" >> $LOG
  $N ./psph_gpu -selftest quick -j $J > logs/selftest_gpu.log 2>&1
  echo "=== selftest quick (GPU) done $(date): $(grep -E 'selftest:' logs/selftest_gpu.log)" >> $LOG
fi
# ---- 2. differential tests
# diff_case TAG CPU_BINARY GPU_BINARY ARGS...: run both binaries with ARGS (each limited to 900 s) and compare as above.
diff_case() {   # tag binary_cpu binary_gpu args...
  tag=$1; bc=$2; bg=$3; shift 3
  echo "--- $tag: $*  start $(date)" >> $LOG
  t0=$(date +%s); $N timeout 900 ./$bc "$@" -j $J -p 0 > logs/$tag.cpu.out 2> logs/$tag.cpu.err; t1=$(date +%s)
  $N timeout 900 ./$bg "$@" -j $J -p 0 > logs/$tag.gpu.out 2> logs/$tag.gpu.err; t2=$(date +%s)
  echo "cpu-wall $((t1-t0)) s" >> logs/$tag.cpu.err; echo "gpu-wall $((t2-t1)) s" >> logs/$tag.gpu.err
  sort logs/$tag.cpu.out > logs/$tag.cpu.sorted; sort logs/$tag.gpu.out > logs/$tag.gpu.sorted
  if cmp -s logs/$tag.cpu.sorted logs/$tag.gpu.sorted; then r="SOLUTIONS IDENTICAL ($(wc -l < logs/$tag.gpu.sorted) bases)"; else r="*** SOLUTION SETS DIFFER ***"; fi
  xc=$(grep -o 'X-pass: [0-9]*' logs/$tag.cpu.err); xg=$(grep -o 'X-pass: [0-9]*' logs/$tag.gpu.err)
  if [ "$xc" = "$xg" ]; then x="X-pass identical ($xc)"; else x="*** X-PASS DIFFERS: cpu $xc gpu $xg ***"; fi
  echo "    $r; $x" >> $LOG
  echo "    cpu: $(grep 'leaf candidates' logs/$tag.cpu.err | cut -c8-)" >> $LOG
  echo "    gpu: $(grep 'leaf candidates' logs/$tag.gpu.err | cut -c8-)" >> $LOG
  echo "    cpu: $(grep 'probes:' logs/$tag.cpu.err)   gpu: $(grep 'probes:' logs/$tag.gpu.err)" >> $LOG
  echo "    $(grep -E 'cpu-wall|gpu-wall' logs/$tag.cpu.err logs/$tag.gpu.err | sed 's/.*err://' | tr '\n' ' ')" >> $LOG
  echo "    $(grep '^\[psph\] wall' logs/$tag.cpu.err | cut -c8-80)  ||  $(grep '^\[psph\] wall' logs/$tag.gpu.err | cut -c8-80)" >> $LOG
  grep -h '^\[gpu\] GPU time' logs/$tag.gpu.err | sed 's/^/    /' >> $LOG
  grep -h 'PARANOID\|FATAL\|INTERNAL' logs/$tag.gpu.err logs/$tag.cpu.err | sed 's/^/    /' >> $LOG
  echo "--- $tag done $(date)" >> $LOG
}
# work-item sub-ranges (-d/-i) keep every run under ~15 min on the shared orin1 CPU; the ranges contain the known
# extremal prefixes: (9,7): item 147 = {1,7,30} at depth 3 (201 items); (14,6): item 508 = {1,11,49} at depth 3
# (651 items); (9,8): item 1199 = {1,3,14,46} at depth 4 (11003 items); (8,8): 6332 items at depth 4
diff_case h12k6_5118    psph_cpu   psph_gpu   -h 12 -k 6 -t 5118
diff_case h9k5_797_16   psph_cpu16 psph_gpu16 -h 9 -k 5 -t 797
diff_case h9k8_5521_i   psph_cpu   psph_gpu   -h 9 -k 8 -t 5521 -d 4 -i 1195:1205
diff_case h9k8_5600_i   psph_cpu   psph_gpu   -h 9 -k 8 -t 5600 -d 4 -i 10900:11003
diff_case h9k7_3191_i   psph_cpu   psph_gpu   -h 9 -k 7 -t 3191 -d 3 -i 144:152
diff_case h9k7_3192_i   psph_cpu   psph_gpu   -h 9 -k 7 -t 3192 -d 3 -i 144:152
diff_case h14k6_9748_i  psph_cpu   psph_gpu   -h 14 -k 6 -t 9748 -d 3 -i 440:600
diff_case h8k8_3300_i   psph_cpu   psph_gpu   -h 8 -k 8 -t 3300 -d 4 -i 3000:3002
diff_case h8k8_3485_i   psph_cpu   psph_gpu   -h 8 -k 8 -t 3485 -d 4 -i 3000:3002
# -s2 0: no deep holes on either side -> every counter must agree exactly
diff_case s2z_h12k6     psph_cpu   psph_gpu   -h 12 -k 6 -t 5118 -s2 0
diff_case s2z_h9k8_i    psph_cpu   psph_gpu   -h 9 -k 8 -t 5600 -d 4 -i 10900:11003 -s2 0
diff_case s2z_h9k5_16   psph_cpu16 psph_gpu16 -h 9 -k 5 -t 797 -s2 0
# int64 kernel path forced + PARANOID host re-check (small cases: the host re-check is sequential and slow)
# par_case TAG REF_SORTED ARGS...: run psph_gpu with PSPH_GPU_INT64=1 PSPH_GPU_PARANOID=1 and compare its sorted
# solutions with a reference file produced earlier by the CPU binary (a PARANOID mismatch would also abort with exit 8).
par_case() {  # tag ref_sorted args...
  tag=$1; ref=$2; shift 2
  echo "--- $tag (int64 + paranoid): $*  start $(date)" >> $LOG
  PSPH_GPU_INT64=1 PSPH_GPU_PARANOID=1 $N timeout 900 ./psph_gpu "$@" -j $J -p 0 > logs/$tag.out 2> logs/$tag.err
  sort logs/$tag.out > logs/$tag.sorted
  cmp -s logs/$tag.sorted logs/$ref && echo "    $tag: SOLUTIONS IDENTICAL to $ref" >> $LOG || echo "    $tag: *** DIFFER from $ref ***" >> $LOG
  echo "    gpu: $(grep 'leaf candidates' logs/$tag.err | cut -c8-)" >> $LOG
  grep -h 'PARANOID\|FATAL\|INTERNAL' logs/$tag.err | grep -v 'host re-check' | sed 's/^/    /' >> $LOG
  echo "--- $tag done $(date)" >> $LOG
}
par_case par64_h9k5_s2z  s2z_h9k5_16.cpu.sorted -h 9 -k 5 -t 797 -s2 0
par_case par64_h9k5      h9k5_797_16.cpu.sorted -h 9 -k 5 -t 797
$N timeout 600 ./psph_cpu -h 12 -k 5 -t 2047 -j $J -p 0 > logs/par_h12k5.cpu.out 2> logs/par_h12k5.cpu.err; sort logs/par_h12k5.cpu.out > logs/par_h12k5.cpu.sorted
par_case par64_h12k5     par_h12k5.cpu.sorted   -h 12 -k 5 -t 2047
echo "    cpu: $(grep 'leaf candidates' logs/par_h12k5.cpu.err | cut -c8-)" >> $LOG
echo "=== validate.sh done $(date)" >> $LOG
