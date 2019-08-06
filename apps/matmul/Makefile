BRISBANE=$(HOME)/.local

all: matmul matmul-brisbane

matmul: matmul.cpp
	g++ -g -o $@ $^

matmul-brisbane: matmul-brisbane.cpp
	g++ -g -I$(BRISBANE)/include -o $@ $^ -L$(BRISBANE)/lib64 -lbrisbane -lOpenCL -lpthread -ldl

clean:
	rm -f matmul matmul-brisbane
