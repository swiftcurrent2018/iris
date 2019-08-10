all: brisbane

vecadd.ll: vecadd.c
	clang -S -emit-llvm $^ -Xclang -disable-O0-optnone

brisbane: vecadd.ll
	opt -S -polly-canonicalize $^ -o vecadd.preopt.ll
	opt -load libLLVMBrisbaneX.so -basicaa -brisbane-x -analyze vecadd.preopt.ll -polly-process-unprofitable -polly-use-llvm-names 2>&1 1>/dev/null

polly: vecadd.ll
	opt -S -polly-canonicalize $^ -o vecadd.preopt.ll
	#opt -basicaa -polly-ast -analyze vecadd.preopt.ll -polly-process-unprofitable -polly-use-llvm-names
	opt -basicaa -polly-scops -analyze vecadd.preopt.ll -polly-process-unprofitable -polly-use-llvm-names

clean:
	rm -f vecadd.ll vecadd.bc vecadd.bc.ll vecadd.preopt.ll

