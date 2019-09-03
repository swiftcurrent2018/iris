TEST=test07_policy_register

all: $(TEST) policy_last.so

include ../Makefile.tests

policy_last.so: policy_last.cpp
	g++ -I$(HOME)/work/brisbane-rts/src/runtime -I$(HOME)/work/brisbane-rts/build/src/runtime -fPIC -shared -o $@ $^

