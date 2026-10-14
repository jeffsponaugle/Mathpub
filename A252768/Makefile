# A252768 tool. Needs primesieve >= 11 (brew install primesieve / apt install libprimesieve-dev);
# GMP (brew install gmp / apt install libgmp-dev) is optional and adds independent primality checks.
#
# PREFIX is where primesieve (and GMP) live when they are not in the compiler's default paths:
# the Homebrew prefix on macOS, /usr/local elsewhere.  Override with `make PREFIX=/opt/foo`.
PREFIX  ?= $(shell brew --prefix 2>/dev/null || echo /usr/local)
CFLAGS  ?= -O2 -std=gnu11 -Wall -Wextra -pthread -I$(PREFIX)/include
LDFLAGS ?= -L$(PREFIX)/lib -Wl,-rpath,$(PREFIX)/lib
LDLIBS  ?= -lprimesieve -lm

# enable GMP if the compiler can find gmp.h (any location, not just PREFIX)
HAVE_GMP := $(shell echo '\#include <gmp.h>' | $(CC) $(CFLAGS) -E -x c - >/dev/null 2>&1 && echo yes)
ifeq ($(HAVE_GMP),yes)
CFLAGS  += -DHAVE_GMP
LDLIBS  += -lgmp
endif

a252768: a252768.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test: a252768
	./a252768 selftest

clean:
	rm -f a252768

.PHONY: test clean
