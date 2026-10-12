# Pure-Python verification of the Wilson-number condition from the definition:
#   P(n) = prod_{1<=a<=n, gcd(a,n)=1} a  (mod n^2),   n is a Wilson number iff P(n) + e(n) == 0 (mod n^2),
#   e(n) = +1 iff n in {1,2,4} or n = p^k or 2p^k (p odd prime), else -1.
import sys, math, time
def factor(n):
    f={};d=2
    while d*d<=n:
        while n%d==0: f[d]=f.get(d,0)+1; n//=d
        d+=1
    if n>1: f[n]=f.get(n,0)+1
    return f
def e_of(n):
    if n in (1,2,4): return 1
    f=factor(n)
    if len(f)==1 and 2 not in f: return 1
    if len(f)==2 and f.get(2,0)==1: return 1
    return -1
for n in map(int,sys.argv[1:]):
    t=time.time(); n2=n*n; P=1; g=math.gcd
    for a in range(1,n+1):
        if g(a,n)==1: P=P*a%n2
    print(n, "factors", factor(n), "P(n) mod n^2 =", P, "e(n) =", e_of(n), "->", "WILSON NUMBER" if (P+e_of(n))%n2==0 else "not Wilson", "(%.0f s)"%(time.time()-t), flush=True)
