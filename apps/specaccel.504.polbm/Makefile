BRISBANE=../../install
LDFLAGS=-L$(BRISBANE)/lib -L$(HOME)/install/lib -lbrisbane -lOpenCL -lpthread

all: lbm lbm-brisbane

lbm: lbm.c main.c
	g++ -I . -DSPEC -o $@ $^

lbm-brisbane: lbm-brisbane.c main-brisbane.c
	g++ -I. -I$(BRISBANE)/include -DSPEC -o $@ $^ $(LDFLAGS)

run: lbm
	./$^ 20 reference.dat 0 1 100_100_130_cf_a.of

run-brisbane: lbm-brisbane
	./$^ 20 reference.dat 0 1 100_100_130_cf_a.of

clean:
	rm -f lbm lbm-brisbane
