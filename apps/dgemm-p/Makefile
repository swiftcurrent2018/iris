BRISBANE=$(HOME)/.local

all: dgemm dgemm-brisbane libbrisbane_poly.so

dgemm: dgemm.cpp
	g++ -g -o $@ $^

dgemm-brisbane: dgemm-brisbane.cpp
	g++ -g -I$(BRISBANE)/include -o $@ $^ -L$(BRISBANE)/lib64 -lbrisbane -lOpenCL -lpthread -ldl

libbrisbane_poly.so: kernel-poly.c
	g++ -I$(BRISBANE)/include -fPIC -shared -rdynamic -o $@ $^

clean:
	rm -f dgemm dgemm-brisbane libbrisbane_poly.so

