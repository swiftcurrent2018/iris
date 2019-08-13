all: kernel-poly.h

vecadd.cl.poly.c: vecadd.cl
	-brisbane-clang $^ -- 2>/dev/null

vecadd.cl.poly.ll: vecadd.cl.poly.c
	clang -S -emit-llvm $^ -Xclang -disable-O0-optnone

vecadd.cl.poly.preopt.ll: vecadd.cl.poly.ll
	opt -S -polly-canonicalize $^ -o $@

kernel-poly.h: vecadd.cl.poly.preopt.ll
	opt -load libLLVMBrisbane.so -basicaa -brisbane -analyze $^ -polly-process-unprofitable -polly-use-llvm-names 2>$@ 1>/dev/null

polly: vecadd.cl.poly.preopt.ll
	opt -basicaa -polly-scops -analyze $^ -polly-process-unprofitable -polly-use-llvm-names

cl: vecadd.cl
	clang -Xclang -finclude-default-header -S -emit-llvm -cl-std=CL2.0 vecadd.cl

clean:
	rm -f vecadd.cl.poly.c vecadd.cl.poly.ll vecadd.cl.poly.preopt.ll kernel-poly.h vecadd.ll

