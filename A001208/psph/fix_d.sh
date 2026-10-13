#!/bin/sh
# mathd: the depth-2 chunking of A001209(303) (10 items per chunk) can keep at most 10 of 96 threads busy
# (load average fell to ~10). Stop it (wrapper first) and measure the depth-3 item count for a finer split.
#
# One-off on mathd, Oct 3 2026 ~07:45 PDT.  Order: wrapper jobs10.sh, then the h303k4 run_range.sh, then psph16.  The
# depth-3 count printed at the end (4158140 items, 515 s to generate) led to mathd_jobs11.sh.  Linux only (free -g).
cd ~/A0001208/psph
for p in $(pgrep -f "sh [j]obs10.sh"); do kill $p; done
sleep 1
for p in $(pgrep -f "[r]un_range.sh h303k4"); do kill $p; done
sleep 1
for p in $(pgrep -x psph16); do kill $p; done
sleep 2
echo "left: wrappers=$(pgrep -af '[j]obs[0-9]*.sh' | wc -l) drivers=$(pgrep -af '[r]un_range' | wc -l) psph16=$(pgrep -c -x psph16)"
echo "=== JOB2 stopped $(date): depth-2 chunks of 10 items use only 10 threads; re-splitting at depth 3" >> jobs.log
echo "done chunks at depth 2: $(ls logs/h303k4/*.done | wc -l) (items 0-29); chunk 3 (items 30-39) killed at: $(tail -1 logs/h303k4/chunk_3.err | cut -c1-90)"
free -g | head -2
echo "--- depth-3 item count:"
s=$(date +%s); ./psph16 -h 303 -k 4 -t 71148328 -j 1 -d 3 -i 0:0 2>&1 | grep -E "split depth|processing|LOW" ; e=$(date +%s); echo "gen wall ${e}-${s} = $((e-s)) s"
