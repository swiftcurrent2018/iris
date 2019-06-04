BRISBANE=../../install

all: vecadd vecadd-brisbane

vecadd: vecadd.cpp
	g++ -o $@ $^

vecadd-brisbane: vecadd-brisbane.cpp
	g++ -I$(BRISBANE)/include -o $@ $^ -L$(BRISBANE)/lib -lbrisbane -lOpenCL -lpthread

clean:
	rm -f vecadd vecadd-brisbane
