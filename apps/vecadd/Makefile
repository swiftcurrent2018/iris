BRISBANE=../../install

all: vecadd vecadd-brisbane

vecadd: vecadd.cpp
	g++ -o $@ $^

vecadd-brisbane: vecadd-brisbane.cpp
	g++ -I$(BRISBANE)/include -o $@ $^ -L$(BRISBANE)/lib -lbrisbane -lOpenCL

clean:
	rm -f vecadd vecadd-brisbane
