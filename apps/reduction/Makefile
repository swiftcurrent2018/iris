BRISBANE=../../install

all: reduction reduction-brisbane

reduction: reduction.cpp
	g++ -o $@ $^

reduction-brisbane: reduction-brisbane.cpp
	g++ -I$(BRISBANE)/include -o $@ $^ -L$(BRISBANE)/lib -L$(HOME)/install/lib -lbrisbane -lOpenCL -lpthread

clean:
	rm -f reduction reduction-brisbane
