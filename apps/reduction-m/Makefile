BRISBANE=$(HOME)/.local

all: reduction-m reduction-m-brisbane

reduction-m: reduction-m.cpp
	g++ -o $@ $^

reduction-m-brisbane: reduction-m-brisbane.cpp
	g++ -I$(BRISBANE)/include -o $@ $^ -L$(BRISBANE)/lib64 -lbrisbane -lOpenCL -lpthread -ldl

clean:
	rm -f reduction-m reduction-m-brisbane
