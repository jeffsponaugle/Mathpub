# A003630 conjecture checker. Needs primesieve >= 11 (brew install primesieve / apt install libprimesieve-dev).
#
# PREFIX is where primesieve lives when it is not in the compiler's default paths:
# the Homebrew prefix on macOS, /usr/local elsewhere.  Override with `make PREFIX=/opt/foo`.
PREFIX  ?= $(shell brew --prefix 2>/dev/null || echo /usr/local)
CFLAGS  ?= -O2 -std=gnu11 -Wall -Wextra -pthread -I$(PREFIX)/include
LDFLAGS ?= -L$(PREFIX)/lib -Wl,-rpath,$(PREFIX)/lib
LDLIBS  ?= -lprimesieve

a003630: a003630.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test: a003630
	./a003630 selftest

clean:
	rm -f a003630

.PHONY: test clean
