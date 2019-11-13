BRISBANE=$(HOME)/.local
CC=xlc
CXX=xlC

all: saxpy saxpy-brisbane

saxpy: saxpy.cpp
	$(CXX) -g -o $@ $^

saxpy-brisbane: saxpy-brisbane.cpp
	$(CXX) -g -I$(BRISBANE)/include -o $@ $^ -L$(BRISBANE)/lib64 -lbrisbane -lpthread -ldl

kernel.ptx: kernel.cu
	nvcc -ptx $^

kernel.hip: kernel.hip.cpp
	hipcc --genco -o $@ $^

kernel.openmp.so: kernel.cl.openmp.c
	$(CC) -g -qsmp -fPIC -shared -I. -I$(BRISBANE)/include -o $@ $^
	#$(CC) -g -fopenmp -fPIC -shared -I. -I$(BRISBANE)/include -o $@ $^

clean:
	rm -f saxpy saxpy-brisbane libbrisbane-poly.so
