#!/bin/sh
# mathb wind-down: kill the wrapper so nothing starts after chunk 29; the running driver/psph finish chunk 29 on their own.
# One-off, Oct 3 2026 07:59 PDT (Jeff asked to wind the x86 boxes down); chunk 29 finished at 08:13 PDT and completed
# the A001211(27) search.  Prints the remaining wrappers / driver / psph count and the chunk's last progress line.
cd ~/A001208/psph
for p in $(pgrep -f "sh [j]obs8.sh"); do kill $p; done
sleep 1
echo "=== WIND-DOWN $(date): wrapper killed; chunk 29 (h27k6j) finishes, then the box goes idle (A001210(91) phase 2 cancelled)" >> jobs.log
echo "wrappers left: $(pgrep -af '[j]obs[0-9]*.sh' | wc -l); driver: $(pgrep -af '[r]un_range' | cut -c1-70); psph: $(pgrep -c -x psph)"
tail -1 logs/h27k6j/chunk_0.err | cut -c1-120
