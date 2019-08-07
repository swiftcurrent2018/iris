BRISBANE=$(HOME)/.local

all: dgemm dgemm-brisbane

dgemm: dgemm.cpp
	g++ -g -o $@ $^

dgemm-brisbane: dgemm-brisbane.cpp
	g++ -g -I$(BRISBANE)/include -o $@ $^ -L$(BRISBANE)/lib64 -lbrisbane -lOpenCL -lpthread -ldl

clean:
	rm -f dgemm dgemm-brisbane
