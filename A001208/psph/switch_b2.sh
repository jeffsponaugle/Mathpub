#!/bin/sh
# One-off reconfiguration script (rebalance #7, Oct 3 2026 ~05:40 PDT, run on mathb in its psph directory): replace
# jobs7.sh by jobs8.sh, which adds chunks 25 and 29 (taken from mathd) after chunk 24.
# Pattern: kill the old wrapper only (the running run_range.sh driver and psph keep going), touch .done SKIP markers
# for chunks that move to another box (unless they had already started there), start the new wrapper with nohup,
# log the switch in jobs.log and show the running wrappers / drivers.  Kept for the record of how the chunks moved.
cd ~/A001208/psph
for p in $(pgrep -f "sh [j]obs7.sh"); do kill $p; done
sleep 1
nohup sh jobs8.sh > /dev/null 2>&1 < /dev/null &
sleep 2
echo "wrappers: $(pgrep -af '[j]obs[0-9]*.sh' | cut -c1-60 | tr '\n' ' ')"; echo "drivers: $(pgrep -af '[r]un_range' | cut -c1-90)"
echo "=== REBALANCE7 $(date): after chunk 24, chunks 25 and 29 (from mathd) at TGT 186943, then h91k5b" >> jobs.log
tail -1 logs/h27k6h/chunk_2.err | cut -c1-110
