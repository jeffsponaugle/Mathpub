#!/bin/sh
# One-off reconfiguration script (rebalance #6, Oct 3 2026 ~01:10 PDT, run on mathd in its psph directory): replace
# jobs9.sh by jobs10.sh; chunk 13 (already started) stays with the h27k6d driver, chunk 14 moves to the new TGT-186943
# range h27k6i.
# Pattern: kill the old wrapper only (the running run_range.sh driver and psph keep going), touch .done SKIP markers
# for chunks that move to another box (unless they had already started there), start the new wrapper with nohup,
# log the switch in jobs.log and show the running wrappers / drivers.  Kept for the record of how the chunks moved.
cd ~/A0001208/psph
for p in $(pgrep -f "sh [j]obs9.sh"); do kill $p; done
sleep 1
mkdir -p logs/h27k6i
for j in 7 8; do
  if [ -f logs/h27k6d/chunk_$j.err ]; then touch logs/h27k6i/chunk_$((j-7)).done; echo "chunk $((j+6)) stays with the h27k6d driver (already started)";
  else touch logs/h27k6d/chunk_$j.done; echo "chunk $((j+6)) moved to h27k6i (TGT 186943)"; fi
done
nohup sh jobs10.sh > /dev/null 2>&1 < /dev/null &
sleep 2
echo "wrappers: $(pgrep -af '[j]obs[0-9]*.sh' | cut -c1-60 | tr '\n' ' ')"; echo "drivers: $(pgrep -af '[r]un_range' | cut -c1-90)"
echo "=== SWITCH2 $(date): remaining n(27,6) chunks at TGT 186943; chunk 25 added from mathb" >> jobs.log
tail -1 logs/h27k6d/chunk_7.err 2>/dev/null | cut -c1-120
