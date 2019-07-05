BRISBANE=../../install

all: reduction-m reduction-m-brisbane

reduction-m: reduction-m.cpp
	g++ -o $@ $^

reduction-m-brisbane: reduction-m-brisbane.cpp
	g++ -I$(BRISBANE)/include -o $@ $^ -L$(BRISBANE)/lib -L$(HOME)/install/lib -lbrisbane -lOpenCL -lpthread

clean:
	rm -f reduction-m reduction-m-brisbane
