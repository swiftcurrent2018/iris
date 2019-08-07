all: opt

vecadd.ll: vecadd.c
	clang -S -emit-llvm $^ -Xclang -disable-O0-optnone

opt: vecadd.ll
	opt -load LLVMBrisbanePoly.so -brisbane-poly < vecadd.ll -o vecadd.bc
	llvm-dis vecadd.bc -o vecadd.bc.ll

polly: vecadd.ll
	opt -S -polly-canonicalize $^ -o vecadd.preopt.ll
	opt -basicaa -polly-ast -analyze vecadd.preopt.ll -polly-process-unprofitable -polly-use-llvm-names
	opt -basicaa -polly-scops -analyze vecadd.preopt.ll -polly-process-unprofitable -polly-use-llvm-names

clean:
	rm -f vecadd.ll vecadd.bc vecadd.bc.ll vecadd.preopt.ll

