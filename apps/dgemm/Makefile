BRISBANE=$(HOME)/.local

all: dgemm dgemm-brisbane

dgemm: dgemm.cpp
	g++ -g -o $@ $^

kernel.ptx: kernel.cu
	nvcc -ptx $^

dgemm-brisbane: dgemm-brisbane.cpp
	g++ -g -I$(BRISBANE)/include -o $@ $^ -L$(BRISBANE)/lib64 -lbrisbane -lpthread -ldl

clean:
	rm -f dgemm dgemm-brisbane libbrisbane-poly.so *.dot *.html
