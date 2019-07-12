BRISBANE=../../install
CFLAGS=-g -I$(BRISBANE)/include
LDFLAGS=-L$(BRISBANE)/lib -L$(HOME)/install/lib -lbrisbane -lOpenCL -pthread

all: epoch epoch-brisbane epoch-brisbane-ver2

epoch: epoch.cpp
	g++ -g -o $@ $^

epoch-brisbane: epoch-brisbane.cpp
	g++ $(CFLAGS) -o $@ $^ -L$(BRISBANE)/lib $(LDFLAGS)

epoch-brisbane-ver2: epoch-brisbane-ver2.cpp
	g++ $(CFLAGS) -o $@ $^ -L$(BRISBANE)/lib $(LDFLAGS)

clean:
	rm -f epoch epoch-brisbane epoch-brisbane-ver2
