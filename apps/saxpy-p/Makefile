BRISBANE=$(HOME)/.local

all: saxpy saxpy-brisbane libbrisbane_poly.so

saxpy: saxpy.cpp
	g++ -g -o $@ $^

saxpy-brisbane: saxpy-brisbane.cpp
	g++ -g -I$(BRISBANE)/include -o $@ $^ -L$(BRISBANE)/lib64 -lbrisbane -lOpenCL -lpthread -ldl

libbrisbane_poly.so: kernel-poly.c
	gcc -I$(BRISBANE)/include -fPIC -shared -rdynamic -o $@ $^

kernel-poly.c: kernel.cl
	-brisbane-clang $^ -- 2>/dev/null
	clang -S -I$(HOME)/.local/include -emit-llvm kernel.cl.llvm.c -Xclang -disable-O0-optnone
	opt -S -polly-canonicalize kernel.cl.llvm.ll -o kernel.cl.llvm.preopt.ll
	opt -load libLLVMBrisbane.so -basicaa -brisbane -analyze kernel.cl.llvm.preopt.ll -polly-process-unprofitable -polly-use-llvm-names 2>kernel-poly.h 1>/dev/null

clean:
	rm -f saxpy saxpy-brisbane kernel-poly.h kernel-poly.c libbrisbane_poly.so kernel.cl.llvm.c kernel.cl.llvm.ll kernel.cl.llvm.preopt.ll kernel-llvm.h kernel.ll
