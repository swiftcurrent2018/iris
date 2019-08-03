BRISBANE=../../install

all: eager eager-brisbane eager-brisbane-v2

eager: eager.cpp
	g++ -g -o $@ $^

eager-brisbane: eager-brisbane.cpp
	g++ -g -I$(BRISBANE)/include -o $@ $^ -L$(BRISBANE)/lib -L$(HOME)/install/lib -lbrisbane -lOpenCL -lpthread -ldl

eager-brisbane-v2: eager-brisbane-v2.cpp
	g++ -g -I$(BRISBANE)/include -o $@ $^ -L$(BRISBANE)/lib -L$(HOME)/install/lib -lbrisbane -lOpenCL -lpthread -ldl

clean:
	rm -f eager eager-brisbane eager-brisbane-v2
