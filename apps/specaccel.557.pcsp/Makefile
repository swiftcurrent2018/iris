BRISBANE=$(HOME)/.local
CFLAGS=-g -I. -Wno-write-strings -DSPEC -DSPEC_USE_INNER_SIMD

all: sp sp-brisbane

sp: add.c adi.c error.c exact_rhs.c exact_solution.c initialize.c print_results.c rhs.c set_constants.c sp.c txinvr.c verify.c
	g++ $(CFLAGS) -o $@ $^

sp-brisbane: add-brisbane.c adi-brisbane.c error.c exact_rhs-brisbane.c exact_solution.c initialize-brisbane.c print_results.c rhs-brisbane.c set_constants.c sp-brisbane.c txinvr-brisbane.c verify-brisbane.c
	g++ $(CFLAGS) -I$(BRISBANE)/include -o $@ $^ -L$(BRISBANE)/lib64 -lbrisbane -lOpenCL -lpthread -ldl

run: sp
	./$^ 

run-brisbane: sp-brisbane
	./$^

clean:
	rm -f sp sp-brisbane
