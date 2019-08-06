BRISBANE=$(HOME)/.local

all: reduction reduction-brisbane

reduction: reduction.cpp
	g++ -o $@ $^

reduction-brisbane: reduction-brisbane.cpp
	g++ -I$(BRISBANE)/include -o $@ $^ -L$(BRISBANE)/lib64 -lbrisbane -lOpenCL -lpthread -ldl

clean:
	rm -f reduction reduction-brisbane
