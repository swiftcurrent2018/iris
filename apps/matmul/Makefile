BRISBANE=../../install

all: matmul matmul-brisbane

matmul: matmul.cpp
	g++ -o $@ $^

matmul-brisbane: matmul-brisbane.cpp
	g++ -I$(BRISBANE)/include -o $@ $^ -L$(BRISBANE)/lib -lbrisbane -lOpenCL -lpthread

clean:
	rm -f matmul matmul-brisbane
