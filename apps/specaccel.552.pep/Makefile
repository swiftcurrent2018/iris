BRISBANE=$(HOME)/.local
CFLAGS=-I. -Wno-write-strings -DSPEC -DSPEC_BLOCK_SIZE=4096

all: ep ep-brisbane


ep: c_timers.c ep.c print_results.c randdp.c wtime.c
	g++ $(CFLAGS) -o $@ $^

ep-brisbane: c_timers.c ep-brisbane.c print_results.c randdp.c wtime.c
	g++ $(CFLAGS) -I$(BRISBANE)/include -o $@ $^ -L$(BRISBANE)/lib64 -lbrisbane -lOpenCL -lpthread -ldl

run: ep
	./$^ -o 64x64x64.out 64 64 64 10

run-brisbane: ep-brisbane
	./$^ -o 64x64x64.out-brsbane 64 64 64 10

clean:
	rm -f ep ep-brisbane
