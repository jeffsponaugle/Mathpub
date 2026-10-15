#!/usr/bin/env python3
"""Independent cross-check of a003630.c below a bound X (default 10^8).

Shares no code with the C program:
  * primes from a numpy sieve of Eratosthenes (not primesieve);
  * ymin(s) from ALL pairs x < y of terms (np.minimum.at over the pair sums),
    instead of the C program's downward scan for the largest x;
  * s(N) for every term with a vectorised pass over a(n-1), a(n-2), ...
    (k = 1..K), and a direct triple search for any term that is not settled
    that way;
  * s(N) for every term below 1000 also by itertools.combinations.

Prints the same statistics as `a003630 check X`; with --compare BINARY it runs
the C program on [0, X] and compares every number and the records table.

usage: verify_a003630.py [X] [--compare ./a003630] [--threads T]
"""
import argparse
import itertools
import re
import subprocess
import sys

import numpy as np

S_MAX = 20000          # ymin(s) is tabulated for s <= S_MAX
K = 8                  # previous terms tried by the vectorised pass
INF = np.iinfo(np.int64).max


def parse_num(s):
    s = s.strip()
    if "^" in s:
        b, e = s.split("^")
        return int(b) ** int(e)
    if "e" in s.lower():
        m, e = s.lower().split("e")
        return int(m) * 10 ** int(e)
    return int(s)


def sieve(n):
    """Boolean array is_prime[0..n]."""
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    s[4::2] = False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if s[i]:
            s[i * i::2 * i] = False
    return s


def compute(X):
    isp = sieve(X)
    P = np.flatnonzero(isp).astype(np.int64)
    T = P[(P % 12 == 5) | (P % 12 == 7)]
    n = len(T)
    is_term = np.zeros(X + 1, dtype=bool)
    is_term[T] = True

    # ymin(s) over all pairs x < y of terms with x + y <= S_MAX
    TS = T[T < S_MAX]
    xs, ys = np.meshgrid(TS, TS, indexing="ij")
    sums = xs + ys
    m = (xs < ys) & (sums <= S_MAX)
    ymin = np.full(S_MAX + 1, INF, dtype=np.int64)
    np.minimum.at(ymin, sums[m], ys[m])

    # s(N): first k = 1, 2, ... with s = N - a(n-k) in P2 and ymin(s) < a(n-k)
    s_arr = np.zeros(n, dtype=np.int64)
    k_arr = np.zeros(n, dtype=np.int64)     # largest part is a(n-k)
    done = np.zeros(n, dtype=bool)
    for k in range(1, K + 1):
        idx = np.arange(k, n)
        q = T[idx - k]
        cand = T[idx] - q
        ok = ~done[idx] & (cand <= S_MAX)
        ok[ok] = ymin[cand[ok]] < q[ok]
        s_arr[idx[ok]] = cand[ok]
        k_arr[idx[ok]] = k
        done[idx[ok]] = True

    # anything left: direct triple search with the largest z first
    left = np.flatnonzero(~done)
    for i in left:
        N = int(T[i])
        for j in range(i - 1, -1, -1):
            z = int(T[j])
            s = N - z
            if s >= 2 * z:
                break
            if any(is_term[s - x] and x < s - x < z for x in T[:j] if 2 * x < s):
                s_arr[i] = s
                k_arr[i] = i - j
                done[i] = True
                break

    # brute force for small N with itertools
    small = [int(t) for t in T if t < 1000]
    for i, N in enumerate(small):
        best = 0
        for x, y, z in itertools.combinations(small[:i], 3):
            if x + y + z == N and (best == 0 or x + y < best):
                best = x + y
        if best != s_arr[i]:
            sys.exit(f"brute-force mismatch at N = {N}: {best} vs {s_arr[i]}")

    gaps = np.diff(T)
    g = int(np.argmax(gaps))
    cm = np.maximum.accumulate(s_arr)
    prev = np.concatenate(([0], cm[:-1]))
    rec = np.flatnonzero(s_arr > prev)
    fails = [(int(T[i]), bool(is_term[T[i] - 2])) for i in np.flatnonzero(~done)]
    return {
        "primes": len(P),
        "t5": int(np.count_nonzero(T % 12 == 5)),
        "t7": int(np.count_nonzero(T % 12 == 7)),
        "twin": int(np.count_nonzero(gaps == 2)),
        "maxgap": (int(gaps[g]), int(T[g + 1])),
        "mean": "%.3f" % (s_arr[done].sum() / np.count_nonzero(done)),
        "fails": [f for f, _ in fails],
        "fail_twin": fails,
        "records": [(int(T[i]), int(s_arr[i])) for i in rec],
        "k": {(f"={k}" if k <= K else f">{K}"): int(c) for k, c in
              zip(*np.unique(np.minimum(k_arr[done], K + 1), return_counts=True))},
        "small_checked": len(small),
    }


def parse_c(out):
    """Pull the same statistics out of `a003630 check` output."""
    num = lambda pat: re.search(pat, out, re.M)
    r = {
        "primes": int(num(r"primes in range\s+(\d+)").group(1)),
        "t5": int(num(r"terms \(5 / 7 mod 12\)\s+(\d+) \+").group(1)),
        "t7": int(num(r"terms \(5 / 7 mod 12\)\s+\d+ \+ (\d+)").group(1)),
        "twin": int(num(r"terms with N - 2 a term\s+(\d+)").group(1)),
        "maxgap": tuple(int(v) for v in num(r"largest gap a\(n\)-a\(n-1\)\s+(\d+) \(a\(n\) = (\d+)\)").groups()),
        "mean": num(r"mean s\(N\)\s+([\d.]+)").group(1),
    }
    fl = num(r"C3 fails for\s+\d+ terms?:? ?(.*)$").group(1)
    r["fails"] = [int(v) for v in re.findall(r"(?:^|, )(\d+)", fl)]
    kl = num(r"largest part = a\(n-k\)(.*)$").group(1)
    r["k"] = {op + k: int(c) for op, k, c in re.findall(r"k([=>])(\d+): (\d+)", kl)}
    r["records"] = [(int(a), int(b)) for a, b in
                    re.findall(r"^\s+(\d+)\s+(\d+)\s+\d+ \+ \d+ \+ \d+$", out, re.M)]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("X", nargs="?", default="1e8")
    ap.add_argument("--compare", metavar="BINARY")
    ap.add_argument("--threads", default="4")
    a = ap.parse_args()
    X = parse_num(a.X)

    r = compute(X)
    print(f"verify_a003630.py: terms N <= {X}")
    print(f"  primes                  {r['primes']}")
    print(f"  terms (5 / 7 mod 12)    {r['t5']} + {r['t7']} = {r['t5'] + r['t7']}")
    print(f"  terms with N - 2 a term {r['twin']}")
    print(f"  largest gap             {r['maxgap'][0]} (a(n) = {r['maxgap'][1]})")
    print(f"  mean s(N)               {r['mean']}")
    print(f"  C3 fails for            {r['fail_twin']}")
    print(f"  largest part = a(n-k)   {r['k']}")
    print(f"  itertools brute force   {r['small_checked']} terms below 1000 agree")
    print("  records of s(N):")
    for N, s in r["records"]:
        print(f"  {N:>14} {s:>6}")

    if a.compare:
        out = subprocess.run([a.compare, "check", str(X), "-q", "-t", a.threads],
                             capture_output=True, text=True).stdout
        c = parse_c(out)
        bad = [k for k in c if c[k] != r[k]]
        if bad:
            for k in bad:
                print(f"  MISMATCH {k}: C {c[k]}  python {r[k]}")
            sys.exit(1)
        print(f"  C program agrees on all {len(c)} statistics and all {len(c['records'])} records")


if __name__ == "__main__":
    main()
