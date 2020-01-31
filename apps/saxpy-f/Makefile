BRISBANE=../../install

all: saxpy

saxpy: saxpy.f90
	gfortran -o $@ $^ -I$(BRISBANE)/include/brisbane -L$(BRISBANE)/lib -lbrisbane -lOpenCL -lpthread -ldl

clean:
	rm -f saxpy
