BRISBANE=../../install

all: vecadd vecadd-brisbane

vecadd: vecadd.cpp
	g++ -g -o $@ $^

vecadd-brisbane: vecadd-brisbane.cpp
	g++ -g -I$(BRISBANE)/include -o $@ $^ -L$(BRISBANE)/lib -L$(HOME)/install/lib -lbrisbane -lOpenCL -lpthread -ldl

clean:
	rm -f vecadd vecadd-brisbane
