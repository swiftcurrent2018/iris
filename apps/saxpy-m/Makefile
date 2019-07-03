BRISBANE=../../install

all: saxpy-m saxpy-m-brisbane

saxpy-m: saxpy-m.cpp
	g++ -g -o $@ $^

saxpy-m-brisbane: saxpy-m-brisbane.cpp
	g++ -g -I$(BRISBANE)/include -o $@ $^ -L$(BRISBANE)/lib -L$(HOME)/install/lib -lbrisbane -lOpenCL -lpthread

clean:
	rm -f saxpy-m saxpy-m-brisbane
