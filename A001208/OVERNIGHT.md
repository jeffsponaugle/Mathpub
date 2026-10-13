# Overnight campaign, started 2026-09-30 ~23:10 (M1 Pro only)

## What is running

1. **CPU (10 threads): exact search for n(9,8)** = A053348(9) = A005344(8), target 5419
   (lower bound 5418 found by `heur/hsearch2`: {1,7,23,88,145,638,1142,1610}).
   Script `logs/overnight/run_h9_k8.sh`: 11003 work items (split depth 4) in 28 chunks of 400, each chunk
   a separate `psph -i lo:hi` run; `.done` markers make it resumable (just re-run the script).
   Probe run measured ~5e8 leaf candidates/s and an ETA of 10-11 h that was still rising (later items are
   costlier), so expect 12-20 h total. Progress: `logs/overnight/h9k8_progress.log`; any basis with
   9-range >= 5419 is appended to `logs/overnight/h9k8_solutions.log` as `SOLUTION h=9 k=8 n_h=... : a1 ... a8`.
   Result rule: when all chunks are done, n(9,8) = max n_h over the solutions (there must be at least one,
   since 5418 is attained by a basis; if none appears something is wrong with the target).
   If a chunk shows a solution with n_h > 5418 early, later chunks may be restarted with `TGT=<n_h+1>`
   (faster); the maximum over all chunks is still the answer.

2. **GPU (Metal): A001212 growth measurement**, `psp2gpu -k 20 -t 153` then `-k 21 -t 165` (proof mode:
   must find nothing). Log `logs/overnight/gpu_psp2_k20_k21.log` (progress lines every 16 batches).
   Expected ~3.5e12 nodes (~1.5 h) for k=20 and ~3e13 (~12 h) for k=21 at ~7e8 nodes/s. Purpose: pin the
   growth factor at k=20,21 to firm up the A001212(25) cost estimate (currently 3e17-6e17 nodes).

## Status updates

- 01:05 — GPU k=20 done: 3.57e12 nodes in 6708 s (532M nodes/s), found nothing at T=153 as expected;
  growth x8.4 over k=19. k=21 started 01:04 (expected ~3.4e13 nodes, ~16 h at 600M/s).
- 01:05 — CPU: chunks 0 and 1 (items 0..800) done in 50 and 66 min, 0 solutions (no basis >= 5419 among
  those prefixes yet; the extremal one need not be in the first items). At ~60 min per 400 items the full
  run is ~28 h; expect ~30 % done by 08:00. Resume with the script if interrupted.

- 06:34 — **first solution: n(9,8) >= 5521** via {1, 3, 14, 46, 201, 346, 1135, 1661} (chunk 2, items 800..1200).
  Chunk 2 took 5.4 h (vs 50-66 min for chunks 0-1): 4.2e10 full checks in that prefix region, so the run
  would take days at target 5419. Restarting the remaining chunks with target 5522 and more deep-hole
  targets (-s2) after a short tuning test.

- 08:17 — **PAUSED on request** (SIGSTOP, processes kept in memory): psph chunk 3 (items 1200..1600, 75 min in),
  its runner script, and psp2gpu k=21 (1.03e13 of ~3.4e13 nodes done). Resume with:
  `pkill -CONT -f run_h9_k8.sh; pkill -CONT -f "psph -h 9 -k 8"; pkill -CONT -f "psp2gpu -k"; pkill -CONT -f "sh -c cd /Users/Jeff.Sponaugle/src/math/A001208/psp2gpu"`.
  If the machine is rebooted instead, re-run `logs/overnight/run_h9_k8.sh` (chunks 0-2 are marked done) and
  restart `psp2gpu -k 21 -t 165 -d 15 -b 65536` from scratch.

- 16:36 Oct 1 — resumed on request. The paused local chunk 3 (items 1200-1600) was retired as redundant (mathd had
  completed items 1200-2200 with no basis >= 5600). The Mac now runs ranges released by the long poles instead:
  [10700,11003) and [10000,10500) (from mathb) then [4200,5200) (from mathd), at target 5823, via psph/mac_jobs.sh
  (logs in psph/logs/mac_h9k8c, mac_h9k8b, mac_h9k8d). The GPU k=21 run was resumed with SIGCONT (1.24e13 of ~3.4e13 nodes).

- 19:30 Oct 1 — the resumed GPU k=21 run was killed by the macOS "Impacting Interactivity" watchdog (long dispatches while
  the CPU is saturated). psp2gpu now splits and retries a batch whose command buffer fails instead of exiting; k=21 restarted
  from scratch with -d 15 -b 32768 at ~420M nodes/s (log logs/overnight/gpu_psp2_k21.log, ETA ~22 h). Earlier attempts'
  logs kept as gpu_psp2_k20_k21_attempt*.log.

- 09:10 Oct 2 — Mac PAUSED again on request: its last n(9,8) chunk [4200,4450) (80 % done) was abandoned and re-run on
  mathd (tag h9k8n, ETA ~10:10 PDT); the GPU k=21 run (1.8e13 of ~3.4e13 nodes) is suspended with SIGSTOP:
  resume with `pkill -CONT -f "psp2gpu -k"; pkill -CONT -f "sh -c cd /Users/Jeff.Sponaugle/src/math/A001208/psp2gpu"`.

- 11:04 Oct 2 — Mac resumed: GPU k=21 continues (2.1e13 of ~3.4e13 nodes, ~9 h left); the Mac CPU now runs the tail of the
  A001211(27) search (mathg's chunks 20-29 = items [2000000,2964437) at depth 4, target 176381; psph/mac_jobs5.sh,
  logs in psph/logs/mac_h27k6).

- 14:05 Oct 2 — Mac STOPPED on request: its A001211(27) chunk (82 % done) abandoned, chunks 20-21 handed to mathb's queue;
  GPU k=21 suspended again (SIGSTOP, ~2.5e13 of ~3.4e13 nodes; resume with pkill -CONT as above). The x86 boxes keep running.

## Lower bounds established tonight (all bases verified by direct h-range computation)

| term | bound | basis | source |
|---|---|---|---|
| A053348(9) = A005344(8) = n(9,8) | **= 6082 (PROVEN 2026-10-02 11:18 PDT; all 11003 items searched; basis UNIQUE, Orin re-check done 2026-10-03)** | 1 8 27 88 197 521 1226 1461 | found on mathg 23:00 Oct 1; search complete Oct 2 |
| A001211(27) = n(27,6) | **= 186942 (PROVEN 2026-10-03 08:13 PDT; all 30 chunks + boundary windows audited)** | 1 19 194 1095 7370 27669 | found on mathb chunk 21 Oct 3 00:45; search complete Oct 3 (mathd/mathb/mathg) |
| A001209(303) = n(303,4) | >= 71148327 | 1 228 17657 912312 | formula family (3,A) continued to t=25; hrange verified |
| A001209(304) | >= 72060639 | same basis | formula (4,A) |
| A001209(305) | >= 72972951 | same basis | formula (5,A) |
| A001209(306) | >= 73885263 | same basis | formula (6,A) |
| A001212(25) | >= 228 (known); SA finds nothing higher in 900 s | | heur/sa2 |
| A001212(26) | >= 244 (known); SA finds nothing higher in 600 s | | heur/sa2 |

The heuristic is ~8-10 % short of the optimum on calibration cases ((8,8): 3146 vs 3485; (14,6): 8952 vs 9748),
so the true n(9,8) is probably 5500-5800 and n(27,6) probably 185000-195000.

## Morning checklist

- `cat logs/overnight/h9k8_progress.log` and `logs/overnight/h9k8_solutions.log` (chunks done, solutions).
- `tail -3 logs/overnight/gpu_psp2_k20_k21.log` (k=20 RESULT line, k=21 progress).
- If the CPU run is not finished: `nohup logs/overnight/run_h9_k8.sh &` resumes it; or partition the
  remaining chunks across other machines (`psph -i lo:hi`, deterministic item order).
- Next candidates for the CPU after (9,8): k=6 h=27 (target 176381), k=4 h=303 with `psph16` (target
  71148328, 2000-3300 core-h, best done on the 96-core box), k=5 h=91.

- 15:20 Oct 12 — the suspended GPU k=21 run no longer exists: the Mac was restarted for the macOS 27 upgrade. The k=21
  growth measurement stopped at ~2.5e13 of ~3.4e13 nodes (no result); rerun from scratch if it is still wanted. Note: after
  the upgrade clang cannot link with the default SDK; build with `SDKROOT=$(xcrun --sdk macosx --show-sdk-path) make` (Xcode 26.6 linker vs Command Line Tools 27.0 SDK).
