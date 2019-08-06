BRISBANE=$(HOME)/.local

all: saxpy saxpy-brisbane libbrisbane_poly.so poly_test

saxpy: saxpy.cpp
	g++ -g -o $@ $^

saxpy-brisbane: saxpy-brisbane.cpp
	g++ -g -I$(BRISBANE)/include -o $@ $^ -L$(BRISBANE)/lib64 -lbrisbane -lOpenCL -lpthread -ldl

libbrisbane_poly.so: kernel-poly.c
	g++ -I$(BRISBANE)/include -fPIC -shared -rdynamic -o $@ $^

poly_test: poly_test.cpp
	g++ -g -I$(BRISBANE)/include -o $@ $^ -ldl

clean:
	rm -f saxpy saxpy-brisbane libbrisbane_poly.so poly_test
