BRISBANE=$(HOME)/.local
CFLAGS=-I. -Wno-write-strings -DSPEC

all: cg cg-brisbane

cg: c_timers.c cg.c print_results.c randdp.c wtime.c
	g++ $(CFLAGS) -o $@ $^

cg-brisbane: c_timers.c cg-brisbane.c print_results.c randdp.c wtime.c
	g++ $(CFLAGS) -I$(BRISBANE)/include -o $@ $^ -L$(BRISBANE)/lib64 -lbrisbane -lOpenCL -lpthread -ldl

run: cg
	./$^ 

run-brisbane: cg-brisbane
	./$^

clean:
	rm -f cg cg-brisbane
