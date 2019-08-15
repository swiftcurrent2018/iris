BRISBANE=$(HOME)/.local

all: saxpy saxpy-brisbane

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
	opt -load libLLVMBrisbane.so -basicaa -brisbane -analyze kernel.cl.llvm.preopt.ll -polly-process-unprofitable -polly-use-llvm-names

polly: kernel.cl
	-brisbane-clang $^ -- 2>/dev/null
	clang -S -I$(HOME)/.local/include -emit-llvm kernel.cl.llvm.c -Xclang -disable-O0-optnone
	opt -S -polly-canonicalize kernel.cl.llvm.ll -o kernel.cl.llvm.preopt.ll
	opt -basicaa -polly-scops -analyze kernel.cl.llvm.preopt.ll -polly-process-unprofitable -polly-use-llvm-names

clean:
	rm -f saxpy saxpy-brisbane libbrisbane_poly.so
