BRISBANE=$(HOME)/.local
CFLAGS=-g -I. -Wno-write-strings -DSPEC -DSPEC_USE_INNER_SIMD
LDFLAGS=-L$(BRISBANE)/lib64 -lbrisbane -lOpenCL -lpthread -ldl

all: bt bt-brisbane

bt: add.c adi.c bt.c error.c exact_rhs.c exact_solution.c initialize.c print_results.c rhs.c set_constants.c solve_subs.c verify.c x_solve.c y_solve.c z_solve.c
	g++ $(CFLAGS) -o $@ $^

bt-brisbane: add-brisbane.c adi.c bt-brisbane.c error.c exact_rhs-brisbane.c exact_solution.c initialize-brisbane.c print_results.c rhs-brisbane.c set_constants.c solve_subs.c verify-brisbane.c x_solve-brisbane.c y_solve-brisbane.c z_solve-brisbane.c
	g++ $(CFLAGS) -I$(BRISBANE)/include -o $@ $^ $(LDFLAGS)

run: bt
	./$^ 

run-brisbane: bt-brisbane
	./$^

clean:
	rm -f bt bt-brisbane
