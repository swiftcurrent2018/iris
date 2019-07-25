BRISBANE=../../install

all: saxpy saxpy-brisbane libbrisbane_poly_kernels.so poly_test

saxpy: saxpy.cpp
	g++ -g -o $@ $^

saxpy-brisbane: saxpy-brisbane.cpp
	g++ -g -I$(BRISBANE)/include -o $@ $^ -L$(BRISBANE)/lib -L$(HOME)/install/lib -lbrisbane -lOpenCL -lpthread

libbrisbane_poly_kernels.so: kernel-poly.c
	g++ -I$(BRISBANE)/include -fPIC -shared -rdynamic -o $@ $^

poly_test: poly_test.cpp
	g++ -g -o $@ $^ -ldl

clean:
	rm -f saxpy saxpy-brisbane libbrisbane_poly_kernels.so poly_test
