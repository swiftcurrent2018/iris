BRISBANE=$(HOME)/.local

all: saxpy saxpy-brisbane

saxpy: saxpy.cpp
	g++ -g -o $@ $^

saxpy-brisbane: saxpy-brisbane.cpp
	g++ -g -I$(BRISBANE)/include -o $@ $^ -L$(BRISBANE)/lib64 -lbrisbane -lOpenCL -lpthread -ldl

clean:
	rm -f saxpy saxpy-brisbane
