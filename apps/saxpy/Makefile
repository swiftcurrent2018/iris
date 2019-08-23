BRISBANE=$(HOME)/.local

all: saxpy saxpy-brisbane

saxpy: saxpy.cpp
	g++ -g -o $@ $^

saxpy-brisbane: saxpy-brisbane.cpp
	g++ -g -fopenmp -I$(BRISBANE)/include -o $@ $^ -L$(BRISBANE)/lib64 -lbrisbane -lcuda -lOpenCL -lpthread -ldl 
	#g++ -g -I$(BRISBANE)/include -o $@ $^ -L$(BRISBANE)/lib -L$(BRISBANE)/lib64 -L/opt/rocm/hip/lib -lbrisbane -lOpenCL -lhip_hcc -lpthread -ldl 

kernel.ptx: kernel.cu
	nvcc -ptx $^

kernel.hip: kernel.hip.cpp
	hipcc --genco -o $@ $^

kernel.omp.so: kernel.omp.c
	gcc -g -fopenmp -fPIC -shared -I$(BRISBANE)/include -o $@ $^

clean:
	rm -f saxpy saxpy-brisbane libbrisbane-poly.so
