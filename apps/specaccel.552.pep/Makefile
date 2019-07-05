BRISBANE=../../install
CFLAGS=-I. -Wno-write-strings -DSPEC

all: ep ep-brisbane


ep: c_timers.c ep.c print_results.c randdp.c wtime.c
	g++ $(CFLAGS) -o $@ $^

ep-brisbane: c_timers.c ep-brisbane.c print_results.c randdp.c wtime.c
	g++ $(CFLAGS) -I$(BRISBANE)/include -o $@ $^ -L$(BRISBANE)/lib -lbrisbane -lOpenCL -lpthread

run: ep
	./$^ -o 64x64x64.out 64 64 64 10

run-brisbane: ep-brisbane
	./$^ -o 64x64x64.out-brsbane 64 64 64 10

clean:
	rm -f ep ep-brisbane
