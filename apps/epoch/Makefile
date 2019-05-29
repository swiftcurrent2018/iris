BRISBANE=$(HOME)/work/brisbane/install

all: epoch epoch-brisbane

epoch: epoch.cpp
	g++ -o $@ $^

epoch-brisbane: epoch-brisbane.cpp
	g++ -I$(BRISBANE)/include -o $@ $^ -L$(BRISBANE)/lib -lbrisbane -lOpenCL

clean:
	rm -f epoch epoch-brisbane
