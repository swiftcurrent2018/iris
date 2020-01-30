BRISBANE=$(HOME)/.local
#CC=xlc
#CXX=xlC
CC=gcc
CXX=g++
NVCC=/usr/local/cuda/bin/nvcc

all: kernel.ptx kernel.openmp.so saxpy-brisbane

saxpy-brisbane: saxpy-brisbane.cpp
	$(CXX) -g -I$(BRISBANE)/include -o $@ $^ -L$(BRISBANE)/lib64 -lbrisbane -lpthread -ldl

kernel.ptx: kernel.cu
	$(NVCC) -ptx $^

kernel.hip: kernel.hip.cpp
	hipcc --genco -o $@ $^

kernel.openmp.so: kernel.cl.openmp.c
	$(CC) -g -fopenmp -fPIC -shared -I. -I$(BRISBANE)/include -o $@ $^

clean:
	rm -f kernel.ptx kernel.openmp.so saxpy-brisbane

