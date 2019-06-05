BRISBANE=../../install

all: vecadd vecadd-brisbane

vecadd: vecadd.cpp
	g++ -o $@ $^

vecadd-brisbane: vecadd-brisbane.cpp
	g++ -I$(BRISBANE)/include -o $@ $^ -L$(BRISBANE)/lib -L$(HOME)/install/lib -lbrisbane -lOpenCL -lpthread

clean:
	rm -f vecadd vecadd-brisbane
