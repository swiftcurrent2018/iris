BRISBANE=$(HOME)/.local

all: vecadd vecadd-brisbane

vecadd: vecadd.cpp
	g++ -g -o $@ $^

vecadd-brisbane: vecadd-brisbane.cpp
	g++ -g -I$(BRISBANE)/include -o $@ $^ -L$(BRISBANE)/lib64 -lbrisbane -lcuda -lOpenCL -lpthread -ldl

kernel.ptx: kernel.cu
	nvcc -ptx $^

kernel.hip: kernel.hip.cpp
	hipcc --genco -o $@ $^

clean:
	rm -f vecadd vecadd-brisbane *.dot *.html
