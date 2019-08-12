all: brisbane

vecadd.ll: vecadd.c
	clang -S -emit-llvm $^ -Xclang -disable-O0-optnone

brisbane: vecadd.ll
	opt -S -polly-canonicalize $^ -o vecadd.preopt.ll
	opt -load libLLVMBrisbane.so -basicaa -brisbane -analyze vecadd.preopt.ll -polly-process-unprofitable -polly-use-llvm-names 2>kernel-poly.h 1>/dev/null
	#opt -load libLLVMBrisbane.so -basicaa -brisbane -analyze vecadd.preopt.ll -polly-process-unprofitable -polly-use-llvm-names 2>&1 1>/dev/null

polly: vecadd.ll
	opt -S -polly-canonicalize $^ -o vecadd.preopt.ll
	#opt -basicaa -polly-ast -analyze vecadd.preopt.ll -polly-process-unprofitable -polly-use-llvm-names
	opt -basicaa -polly-scops -analyze vecadd.preopt.ll -polly-process-unprofitable -polly-use-llvm-names

clean:
	rm -f *.ll *.bc *.o kernel-poly.h

