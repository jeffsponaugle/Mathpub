/* wilsonnum_brute.c - direct check of the Wilson-number condition P(n) + e(n) == 0 (mod n^2),
 * P(n) = product of units mod n in [1,n], for n < 2^32 (so n^2 fits in 64 bits). Multithreaded.
 * usage: wilsonnum_brute [-t threads] n [n ...]      -> prints n, P(n) mod n^2, e(n), verdict
 *        wilsonnum_brute [-t threads] -all N        -> scans every n <= N and prints the Wilson numbers
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
typedef unsigned long long u64; typedef unsigned __int128 u128;
static u64 gcd(u64 a, u64 b) { while (b) { u64 t = a % b; a = b; b = t; } return a; }
static int e_of(u64 n) {            /* +1 iff n in {1,2,4}, p^k, 2p^k (p odd) */
    if (n <= 4 && n != 3) return n == 3 ? 1 : 1;
    u64 m = n; int two = 0; while (m % 2 == 0) { m /= 2; two++; }
    if (two >= 2) return (m == 1 && two <= 2) ? 1 : -1;
    /* m odd: must be a prime power */
    if (m == 1) return 1; /* n = 2 */
    u64 q = m; u64 d = 3; while (d * d <= q && q % d) d += 2; u64 p = (d * d <= q) ? d : q;
    while (q % p == 0) q /= p;
    return q == 1 ? 1 : -1;
}
typedef struct { u64 n, lo, hi, prod; } job_t;
static void *worker(void *a) { job_t *j = a; u64 n2 = j->n * j->n, r = 1; for (u64 x = j->lo; x < j->hi; x++) if (gcd(x, j->n) == 1) r = (u64)(((u128)r * x) % n2); j->prod = r; return 0; }
static u64 P_mod_n2(u64 n, int nt) {
    if (n < 100000) nt = 1;
    pthread_t th[256]; job_t jb[256]; u64 n2 = n * n;
    for (int t = 0; t < nt; t++) { jb[t].n = n; jb[t].lo = 1 + n * t / nt; jb[t].hi = 1 + n * (t + 1) / nt; pthread_create(&th[t], 0, worker, &jb[t]); }
    u64 r = 1; for (int t = 0; t < nt; t++) { pthread_join(th[t], 0); r = (u64)(((u128)r * jb[t].prod) % n2); }
    return r;
}
/* range scan: threads take n's dynamically; each n done single-threaded */
static u64 scanN; static volatile u64 next_n = 2; static pthread_mutex_t lk = PTHREAD_MUTEX_INITIALIZER;
static void *scan_worker(void *a) { (void)a; for (;;) { pthread_mutex_lock(&lk); u64 n = next_n; next_n += 64; pthread_mutex_unlock(&lk); if (n > scanN) break;
    for (u64 m = n; m < n + 64 && m <= scanN; m++) { u64 n2 = m * m, r = 1; for (u64 x = 1; x <= m; x++) if (gcd(x, m) == 1) r = (u64)(((u128)r * x) % n2);
        int e = e_of(m); if ((r + (e > 0 ? 1 : n2 - 1)) % n2 == 0) { pthread_mutex_lock(&lk); printf("WILSON %llu\n", m); fflush(stdout); pthread_mutex_unlock(&lk); } } }
    return 0; }
int main(int argc, char **argv) {
    int nt = 8, a = 1; if (argc > 2 && !strcmp(argv[1], "-t")) { nt = atoi(argv[2]); a = 3; }
    if (a < argc && !strcmp(argv[a], "-all")) { scanN = strtoull(argv[a + 1], 0, 10); pthread_t th[256]; for (int t = 0; t < nt; t++) pthread_create(&th[t], 0, scan_worker, 0); for (int t = 0; t < nt; t++) pthread_join(th[t], 0); printf("DONE scan to %llu\n", scanN); return 0; }
    for (; a < argc; a++) { u64 n = strtoull(argv[a], 0, 10); if (n >= (1ULL << 32)) { fprintf(stderr, "n too large\n"); return 1; }
        u64 r = P_mod_n2(n, nt); int e = e_of(n); u64 n2 = n * n; int ok = ((r + (e > 0 ? 1 : n2 - 1)) % n2) == 0;
        printf("%llu  P(n) mod n^2 = %llu  e(n)=%+d  %s\n", n, r, e, ok ? "WILSON NUMBER" : "not a Wilson number"); fflush(stdout); }
    return 0;
}
