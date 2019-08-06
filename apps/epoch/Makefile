BRISBANE=$(HOME)/.local
CFLAGS=-g -I$(BRISBANE)/include
LDFLAGS=-L$(BRISBANE)/lib64 -lbrisbane -lOpenCL -pthread -ldl

all: epoch epoch-brisbane epoch-brisbane-v2

epoch: epoch.cpp
	g++ -g -o $@ $^

epoch-brisbane: epoch-brisbane.cpp
	g++ $(CFLAGS) -o $@ $^ -L$(BRISBANE)/lib $(LDFLAGS)

epoch-brisbane-v2: epoch-brisbane-v2.cpp
	g++ $(CFLAGS) -o $@ $^ -L$(BRISBANE)/lib $(LDFLAGS)

clean:
	rm -f epoch epoch-brisbane epoch-brisbane-v2
