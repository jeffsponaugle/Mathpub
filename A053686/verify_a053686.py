#!/usr/bin/env python3
"""
Independent check of the a053686 tool (OEIS A053686: record prime gaps that
repeat before the next record).

Shares no code with the C program: its own copy of the maximal-gap table
(A005250 / A002386), deterministic Miller-Rabin for 64-bit numbers, and a
numpy sieve.

usage: verify_a053686.py P G           check that P and P+G are consecutive primes and
                                       say whether the gap is a repeat of the record of
                                       its interval (i.e. a term of A053686)
       verify_a053686.py sieve [N]     sieve to N (default 2e8), decide every record
                                       interval below N, compare with OEIS and with
                                       './a053686 scan 0 N -c 1e6 -q -H'
       verify_a053686.py table         check the table: each entry a prime gap of the
                                       listed size, A133788 = first occurrences of the terms
       verify_a053686.py               table + sieve 2e8
"""
import re
import subprocess
import sys

# (record gap, prime after which it first occurs): A005250(n), A002386(n), n = 1..80 (all records below 2^64)
REC = [
    (1, 2), (2, 3), (4, 7), (6, 23), (8, 89), (14, 113), (18, 523), (20, 887), (22, 1129), (34, 1327),
    (36, 9551), (44, 15683), (52, 19609), (72, 31397), (86, 155921), (96, 360653), (112, 370261),
    (114, 492113), (118, 1349533), (132, 1357201), (148, 2010733), (154, 4652353), (180, 17051707),
    (210, 20831323), (220, 47326693), (222, 122164747), (234, 189695659), (248, 191912783),
    (250, 387096133), (282, 436273009), (288, 1294268491), (292, 1453168141), (320, 2300942549),
    (336, 3842610773), (354, 4302407359), (382, 10726904659), (384, 20678048297), (394, 22367084959),
    (456, 25056082087), (464, 42652618343), (468, 127976334671), (474, 182226896239),
    (486, 241160624143), (490, 297501075799), (500, 303371455241), (514, 304599508537),
    (516, 416608695821), (532, 461690510011), (534, 614487453523), (540, 738832927927),
    (582, 1346294310749), (588, 1408695493609), (602, 1968188556461), (652, 2614941710599),
    (674, 7177162611713), (716, 13829048559701), (766, 19581334192423), (778, 42842283925351),
    (804, 90874329411493), (806, 171231342420521), (906, 218209405436543), (916, 1189459969825483),
    (924, 1686994940955803), (1132, 1693182318746371), (1184, 43841547845541059),
    (1198, 55350776431903243), (1220, 80873624627234849), (1224, 203986478517455989),
    (1248, 218034721194214273), (1272, 305405826521087869), (1328, 352521223451364323),
    (1356, 401429925999153707), (1370, 418032645936712127), (1442, 804212830686677669),
    (1476, 1425172824437699411), (1488, 5733241593241196731), (1510, 6787988999657777797),
    (1526, 15570628755536096243), (1530, 17678654157568189057), (1550, 18361375334787046697),
]
KNOWN = [2, 4, 6, 14, 34, 36, 52, 86, 132, 154, 250, 336, 906]                  # A053686
A133788 = [3, 7, 23, 113, 1327, 9551, 19609, 155921, 1357201, 4652353, 387096133, 3842610773,
           218209405436543]
OEIS_DECIDED = 63       # records 1..63 are decided in the OEIS (A053686 rev. #28, Sep 17 2026:
                        # a(13) = 906, a(14) >= 1132)
HK = 100                # histogram depth of the tool: gaps g_n - 2k, k < HK
fails = 0


def check(ok, msg):
    global fails
    print(("ok   " if ok else "FAIL ") + msg)
    if not ok:
        fails += 1


def parse_num(s):
    s = s.replace(",", "")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)e(\d+)", s)
    if m:
        v = float(m.group(1)) * 10 ** int(m.group(2))
        return int(round(v))
    m = re.fullmatch(r"(\d+)\^(\d+)", s)
    if m:
        return int(m.group(1)) ** int(m.group(2))
    m = re.fullmatch(r"r(\d+)", s)
    if m:
        return REC[int(m.group(1)) - 1][1]
    return int(s)


def is_prime(n):
    """deterministic for n < 2^64 (bases of Sinclair / Feitsma-Galway)"""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        a %= n
        if a == 0:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def next_prime(n):
    m = n + 1 if n % 2 == 0 else n + 2
    while not is_prime(m):
        m += 2
    return m


def rec_index(x):
    """i with REC[i].p <= x < REC[i+1].p"""
    lo, hi = 0, len(REC) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if REC[mid][1] <= x:
            lo = mid
        else:
            hi = mid - 1
    return lo


def rec_end(i):
    return REC[i + 1][1] if i + 1 < len(REC) else None


def expected_member(n):
    """what OEIS says about record n (1-based): True / False / None (undecided)"""
    if n <= OEIS_DECIDED:
        return REC[n - 1][0] in KNOWN
    return None


# ---------------------------------------------------------------- table

def cmd_table():
    ok = True
    for i, (g, p) in enumerate(REC):
        if i and (g <= REC[i - 1][0] or p <= REC[i - 1][1]):
            ok = False
        if not is_prime(p) or next_prime(p) != p + g:
            ok = False
            print(f"     entry #{i + 1} ({g} after {p}) is not a prime gap of that size")
    check(ok, f"all {len(REC)} table entries are increasing prime gaps of the listed size")
    firsts = [p for g, p in REC if g in KNOWN]
    check(firsts == A133788, f"the first occurrences of the {len(KNOWN)} terms are A133788")
    terms = [n for n, (g, p) in enumerate(REC[:OEIS_DECIDED], 1) if g in KNOWN]
    check([REC[n - 1][0] for n in terms] == KNOWN,
          f"the {len(KNOWN)} terms are records {','.join(map(str, terms))}")


# ---------------------------------------------------------------- P G

def cmd_pair(p, g):
    check(is_prime(p), f"{p} is prime")
    q = next_prime(p)
    check(q == p + g, f"the next prime after {p} is {q}, gap {q - p} (claimed {g})")
    i = rec_index(p)
    n, G, P = i + 1, REC[i][0], REC[i][1]
    end = rec_end(i)
    print(f"     record interval #{n}: gap {G} first after {P}, next record after {end if end else '2^64 or beyond'}")
    if p == P:
        check(q - p == G, f"this is the first occurrence of record #{n} itself")
    elif q - p == G:
        exp = expected_member(n)
        tag = ("= OEIS" if exp else "DIFFERS FROM OEIS") if exp is not None else "NEW"
        print(f"     REPEAT: {G} occurs again before the next record => {G} is a term of A053686  [{tag}]")
        check(exp is not False, f"OEIS does not claim record #{n} is non-repeating")
    elif q - p > G:
        check(False, f"gap {q - p} exceeds the record {G} of its interval (unlisted maximal gap?)")
    else:
        print(f"     not a repeat: the record of this interval is {G}, this gap is short by {G - (q - p)}")


# ---------------------------------------------------------------- sieve

def primes_below(n):
    import numpy as np
    sieve = np.ones(n // 2 + 1, dtype=np.bool_)          # index i <-> 2i+1
    sieve[0] = False
    for i in range(1, int(n ** 0.5) // 2 + 1):
        if sieve[i]:
            p = 2 * i + 1
            sieve[p * p // 2::p] = False
    odd = 2 * np.flatnonzero(sieve) + 1
    return np.concatenate(([2], odd[odd < n])).astype(np.int64)


def analyse(N):
    """everything the tool reports for the gaps after the primes below N"""
    import numpy as np
    pr = primes_below(N + 100000)
    n_below = int(np.searchsorted(pr, N))            # primes < N
    p = pr[:n_below]
    q = pr[1:n_below + 1]
    g = q - p
    tab = [r for r in REC if r[1] < 2 ** 63]            # numpy int64; enough for any sieve below 2^63
    P = np.array([r[1] for r in tab], dtype=np.int64)
    G = np.array([r[0] for r in tab], dtype=np.int64)
    idx = np.searchsorted(P, p, side="right") - 1
    at_record = p == P[idx]
    anomalies = int(np.count_nonzero(g[at_record] != G[idx][at_record]))
    inner = ~at_record
    d = G[idx][inner] - g[inner]
    anomalies += int(np.count_nonzero(d < 0))
    recseen = sorted(set((idx[at_record] + 1).tolist()))
    hist = {}
    sel = (d >= 0) & (d // 2 < HK)
    keys = (idx[inner][sel] + 1) * 10000 + (G[idx][inner][sel] - d[sel])
    for k, c in zip(*np.unique(keys, return_counts=True)):
        hist[(int(k) // 10000, int(k) % 10000)] = int(c)
    reps = {}
    for n_, pp in zip((idx[inner][d == 0] + 1).tolist(), p[inner][d == 0].tolist()):
        reps.setdefault(n_, []).append(pp)
    return {"primes": n_below, "anomalies": anomalies, "recseen": recseen, "hist": hist, "reps": reps}


def run_tool(tool, N):
    out = subprocess.run([tool, "scan", "0", str(N), "-c", "1000000", "-q", "-H"],
                         capture_output=True, text=True, check=True).stdout
    res = {"hist": {}, "reps": {}, "recseen": []}
    for line in out.splitlines():
        m = re.match(r"hist (\d+) (\d+) (\d+)$", line)
        if m:
            res["hist"][(int(m.group(1)), int(m.group(2)))] = int(m.group(3))
            continue
        m = re.match(r"REPEAT\s+#(\d+)\s+gap\s+(\d+)\s+after (\d+)", line)
        if m:
            res["reps"].setdefault(int(m.group(1)), []).append(int(m.group(3)))
            continue
        m = re.match(r"RECORD\s+#(\d+)", line)
        if m:
            res["recseen"].append(int(m.group(1)))
            continue
        m = re.match(r"(\d+) primes in \[0, (\d+)\)", line)
        if m:
            res["primes"] = int(m.group(1))
            continue
        m = re.match(r"\d+ record gaps? in the range.*; (\d+) anomal", line)
        if m:
            res["anomalies"] = int(m.group(1))
    res["recseen"].sort()
    return res


def cmd_sieve(N, tool):
    mine = analyse(N)
    print(f"python: {mine['primes']} primes below {N}, {len(mine['recseen'])} records seen, "
          f"{mine['anomalies']} anomalies, {len(mine['hist'])} histogram entries")
    check(mine["anomalies"] == 0, "no gap exceeds the record of its interval, every record is where A002386 says")
    for i in range(len(REC)):
        end = rec_end(i)
        if end is None or end > N:
            break
        n, g = i + 1, REC[i][0]
        reps = mine["reps"].get(n, [])
        exp = expected_member(n)
        got = bool(reps)
        where = f", first after {reps[0]}" if reps else ""
        if exp is None:
            print(f"     record #{n} = {g}: {'REPEATS' if got else 'does not repeat'}{where} (undecided in OEIS)")
        else:
            check(got == exp, f"record #{n} = {g} {'repeats' if got else 'does not repeat'}{where} "
                              f"(OEIS: {'term' if exp else 'not a term'})")
        for pp in reps[:3]:
            check(next_prime(pp) == pp + g, f"     Miller-Rabin: the prime after {pp} is {pp + g}")
    try:
        tr = run_tool(tool, N)
    except (OSError, subprocess.CalledProcessError) as e:
        check(False, f"could not run {tool}: {e}")
        return
    what = f"{tool} scan 0 {N} -c 1e6"
    for key in ("primes", "anomalies", "recseen", "hist", "reps"):
        same = tr.get(key) == mine[key]
        check(same, f"{what}: {key} agrees" if same else f"{what}: {key} differs: tool {str(tr.get(key))[:200]} vs python {str(mine[key])[:200]}")


def main():
    args = sys.argv[1:]
    tool = "./a053686"
    if "--tool" in args:
        k = args.index("--tool")
        tool = args[k + 1]
        del args[k:k + 2]
    if not args:
        cmd_table()
        cmd_sieve(200000000, tool)
    elif args[0] == "table":
        cmd_table()
    elif args[0] == "sieve":
        cmd_sieve(parse_num(args[1]) if len(args) > 1 else 200000000, tool)
    elif len(args) == 2:
        cmd_pair(parse_num(args[0]), parse_num(args[1]))
    else:
        print(__doc__)
        sys.exit(2)
    print("verification " + ("FAILED" if fails else "passed"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
