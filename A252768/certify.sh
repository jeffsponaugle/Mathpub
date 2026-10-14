#!/bin/sh
# certify.sh P [FILE]
#
# Prove the primality of the power sums S_k(P) printed on the hit line "p = P ..." of
# a scan output file (default scan_1e14.txt) with PARI/GP's isprime, which for numbers
# above 2^64 runs an APR-CL / ECPP proof rather than a probable-prime test.  Prints one
# line per sum and the certified depth of P.  Needs gp (apt install pari-gp).
P=$1
F=${2:-scan_1e14.txt}
[ -z "$P" ] && { echo "usage: certify.sh P [FILE]"; exit 2; }
line=$(grep "^p = $P " "$F") || { echo "no hit line for p = $P in $F"; exit 1; }
# the hit line: p = P  n = d  S_1..S_d = v1 v2 ... vd  S_(d+1) = v (composite)   [or a bracketed note]
sums=$(printf '%s\n' "$line" | sed -e 's/.*S_1\.\.S_[0-9]* =//' -e 's/S_[0-9]* = //' -e 's/(composite)//' -e 's/\[.*\]//')
vec=$(printf '%s\n' "$sums" | tr -s ' ' '\n' | grep -v '^$' | paste -sd, -)
gp -q <<GP
v = [$vec];
d = 0;
for (k = 1, #v, r = isprime(v[k]); print("S_", k, " = ", v[k], " : ", if (r, "prime (proven)", "composite")); if (r && d == k - 1, d = k));
print("depth(", $P, ") = ", d, "  certified with PARI/GP ", version());
GP
