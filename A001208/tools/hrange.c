/* hrange.c -- compute the h-range of a postage stamp basis.
 * usage: hrange h a1 a2 ... ak      (prints largest n with 1..n all sums of <= h elements)
 *        hrange -f file             (file lines: h a1 a2 ... ak ; prints "h k n a1..ak" per line)
 * Method: min-stamp-count DP (unbounded knapsack) over [0, h*ak + 1]. uint16 counts (h up to 65000).
 *
 * Build: clang -O2 -std=gnu11 -o hrange hrange.c        (plain C99, any compiler will do)
 *
 * THEORY OF OPERATIONS
 * ====================
 *
 * 1. What is computed
 *    For a basis A = {a_1 = 1 < a_2 < ... < a_k} and a stamp count h, the h-range n_h(A) is the largest N
 *    such that every integer 1..N is a sum of at most h elements of A (repetition allowed).  The postage
 *    stamp sequences tabulate its maximum over all k-element bases: rows A001212..A001216, A005342,
 *    A005343, A005344, A075060 (fixed h) and columns A001208..A001211, A053346, A053348 (fixed k).
 *    hrange is the project's independent verifier: every basis reported by the exhaustive searches, by
 *    the heuristics in heur/ and taken from the literature tables was re-checked with it.
 *    Example:  hrange 9 1 8 27 88 197 521 1226 1461   prints 6082   (n(9,8) = A053348(9) = A005344(8)).
 *
 * 2. Mathematics
 *    Let m[x] = the minimum number of elements of A that sum to x (infinite if there is none):
 *        m[0] = 0,   m[x] = 1 + min{ m[x - a] : a in A, a <= x }.
 *    Then n_h(A) = (the first x >= 1 with m[x] > h) - 1.  A sum of at most h elements is at most h*a_k,
 *    so x = h*a_k + 1 is never representable: the table [0, h*a_k + 1] always contains the first gap and
 *    the result is never truncated, provided a_k (the last argument) really is the largest element.
 *
 * 3. Algorithm
 *    a) lim = h*a_k + 1; c[0..lim] = 0xFFFF ("unreachable"), c[0] = 0.
 *    b) For each element a_i, for x = a_i .. lim in increasing order: c[x] = min(c[x], c[x - a_i] + 1).
 *       Increasing x lets the same element be used again (unbounded knapsack); doing the elements one
 *       after another gives the minimum over all multisets of elements.
 *    c) Scan n upward from 0 while c[n+1] <= h.
 *    Time O(k * h * a_k); memory 2 * (h*a_k + 2) bytes (for example h = 303, a_k = 912312: about 553 MB).
 *
 * 4. Counter width (why uint16)
 *    0xFFFF marks "unreachable".  The +1 is computed in unsigned int, so 0xFFFF + 1 = 65536 neither wraps
 *    to 0 nor beats any stored value; stored counts are therefore <= 65534 and exact whenever the true
 *    count is <= 65534.  Hence the test c[x] <= h is exact for h <= 65534 ("h up to 65000" above is the
 *    round safe figure).  An earlier version used 8-bit counts: there the marker 0xFF is the same as a
 *    count of 255, so results were wrong for h >= 255, which is why the counts are uint16 now.
 *
 * 5. Command line and input format
 *    hrange h a1 ... ak   h >= 1; elements in increasing order (without a1 = 1 the result is 0); at most
 *                         64 elements.
 *    hrange -f file       one basis per line, "h a1 ... ak", integers separated by white space.  A line
 *                         that does not start with an integer, or has no element after h, is skipped
 *                         silently (so blank lines and "#" comments are fine).  At most 64 elements and
 *                         4095 characters per line.
 *
 * 6. Output
 *    stdout: argv mode prints "n"; file mode prints "h k n a1 ... ak" for each processed line, flushed
 *    line by line.  stderr: the usage line, or "open: <reason>" when the file cannot be opened.
 *    Exit status 0, or 1 for those two errors.
 *
 * 7. Why the result is exact
 *    It is a plain integer dynamic program with no pruning and no sampling, exact within the limits of
 *    sections 2 and 4.  It is a short stand-alone program, which is what makes it a useful independent
 *    check of the search programs.
 *
 * 8. Limits and caveats (none of these are checked by the code)
 *    - Only the LAST element sizes the table: if it is not the largest, the result can be cut off at
 *      h*a_k + 1 and is then wrong.  Give the elements in increasing order.
 *    - At most 64 elements (long a[64]); more overflow the array, in both modes.
 *    - 1 <= h <= 65534 is assumed; h <= 0 gives 0 or a huge, failing (unchecked) allocation.
 *    - Elements must be positive: 0 is ignored, a negative element indexes outside the table.
 *    - malloc() failure is not checked.  Lines longer than 4095 characters are split by fgets() and the
 *      tail is parsed as a separate line.  The -f file is never fclose()d (harmless, the process exits).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/*
 * hrange -- h-range n_h(A) of the basis a[0..k-1] by the min-stamp-count DP.
 *
 * Purpose: the whole computation (THEORY sections 2-4): fill the table of minimal stamp counts over
 *          [0, h*a[k-1] + 1], then return the length of the initial run of values needing <= h stamps.
 * Inputs:  h = stamps per sum, 1..65534; k = number of elements, >= 1 (both callers guarantee it);
 *          a = the elements, positive and in increasing order (a[0] = 1 for a nonzero result).
 *          Only a[k-1] is used to size the table.
 * Outputs: returns the largest n such that 1..n are all sums of at most h elements (0 if 1 is not).
 *          Allocates and frees a table of h*a[k-1] + 2 uint16_t entries; prints nothing.
 * Errors:  malloc() failure is not checked (NULL dereference).  An unsorted input whose last element is
 *          not the largest can give a truncated (wrong) result.  Nonpositive h or elements are not
 *          rejected (see header section 8).
 */
static long hrange(int h, int k, const long *a) {
    /* h*a_k + 1 can never be a sum of <= h elements, so the scan below always stops inside the table */
    long lim = (long)h * a[k - 1] + 1;
    uint16_t *c = malloc((lim + 1) * sizeof(uint16_t));
    /* c[x] = minimal number of elements summing to x; 0xFFFF = not reachable (yet) */
    for (long i = 0; i <= lim; i++) c[i] = 0xFFFF;
    c[0] = 0;
    for (int i = 0; i < k; i++) {
        long ai = a[i];
        /* increasing x: c[x - ai] may already use ai, so ai can be used any number of times */
        for (long x = ai; x <= lim; x++) {
            /* the sum is kept in an unsigned int: 0xFFFF + 1 = 65536 is never < c[x], so "unreachable"
             * stays unreachable (stored in a uint16_t first, it would wrap to 0) */
            unsigned v = c[x - ai] + 1u;
            if (v < c[x]) c[x] = (uint16_t)v;
        }
    }
    /* first gap: smallest x >= 1 with c[x] > h; the range is one less */
    long n = 0;
    while (n + 1 <= lim && c[n + 1] <= h) n++;
    free(c);
    return n;
}

/*
 * main -- command-line front end: one basis from argv, or one basis per line of a file (-f).
 *
 * Purpose: parse the input, call hrange(), print the result(s) (formats in header section 6).
 * Inputs:  argv = "h a1 ... ak" or "-f file".
 * Outputs: stdout: "n" (argv mode) or "h k n a1 ... ak" per processed line (file mode, flushed per line).
 *          Exit status 0; 1 with the usage line on stderr, or with perror("open") if the file cannot be
 *          opened.
 * Errors:  more than 64 elements overflow a[64] (unchecked); atoi()/atol() turn garbage into 0 without a
 *          message; file lines longer than 4095 characters are split; the file is never fclose()d.
 */
int main(int argc, char **argv) {
    /* file mode: "-f file" (with fewer than 3 arguments "-f" falls through to the usage message) */
    if (argc >= 3 && !strcmp(argv[1], "-f")) {
        FILE *f = fopen(argv[2], "r");
        if (!f) { perror("open"); return 1; }
        char line[4096];
        while (fgets(line, sizeof line, f)) {
            long a[64]; int k = 0; int h;
            char *p = line, *e;
            /* h first: a line without a leading integer (blank, comment) is skipped; then read integers
             * until the first token that is not one (no bound check against a[64]) */
            h = (int)strtol(p, &e, 10); if (e == p) continue; p = e;
            for (;;) { long v = strtol(p, &e, 10); if (e == p) break; a[k++] = v; p = e; }
            if (k == 0) continue;
            long n = hrange(h, k, a);
            printf("%d %d %ld", h, k, n);
            for (int i = 0; i < k; i++) printf(" %ld", a[i]);
            printf("\n");
            fflush(stdout);
        }
        return 0;
    }
    /* argv mode: h and at least one element; no check on the count (<= 64) or on the values */
    if (argc < 3) { fprintf(stderr, "usage: hrange h a1 ... ak | hrange -f file\n"); return 1; }
    int h = atoi(argv[1]); long a[64]; int k = 0;
    for (int i = 2; i < argc; i++) a[k++] = atol(argv[i]);
    printf("%ld\n", hrange(h, k, a));
    return 0;
}
