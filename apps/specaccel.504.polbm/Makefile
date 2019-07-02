BRISBANE=../../install

all: polbm polbm-brisbane

polbm: lbm.c main.c
	g++ -I . -DSPEC -o $@ $^

polbm-brisbane: lbm.c main-brisbane.c 
	g++ -I. -I$(BRISBANE)/include -DSPEC -o $@ $^ -L$(BRISBANE)/lib -lbrisbane -lOpenCL -lpthread

run: polbm
	./$^ 20 reference.dat 0 1 100_100_130_cf_a.of

run-brisbane: polbm-brisbane
	./$^ 20 reference.dat 0 1 100_100_130_cf_a.of

clean:
	rm -f polbm polbm-brisbane
