BRISBANE=../../install

all: matmul matmul-brisbane

matmul: matmul.cpp
	g++ -g -o $@ $^

matmul-brisbane: matmul-brisbane.cpp
	g++ -g -I$(BRISBANE)/include -o $@ $^ -L$(BRISBANE)/lib -L$(HOME)/install/lib -lbrisbane -lOpenCL -lpthread

clean:
	rm -f matmul matmul-brisbane
