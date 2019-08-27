BRISBANE=$(HOME)/.local

all: saxpy saxpy-brisbane

saxpy: saxpy.cpp
	g++ -g -o $@ $^

saxpy-brisbane: saxpy-brisbane.cpp
	g++ -g -I$(BRISBANE)/include -o $@ $^ -L$(BRISBANE)/lib -lbrisbane -lpthread -ldl

kernel.ptx: kernel.cu
	nvcc -ptx $^

kernel.hip: kernel.hip.cpp
	hipcc --genco -o $@ $^

kernel.openmp.so: kernel.cl.openmp.h brisbane/kernel.cl.openmp.c
	gcc -g -fopenmp -fPIC -shared -I$(BRISBANE)/include -o $@ $^

clean:
	rm -f saxpy saxpy-brisbane libbrisbane-poly.so
