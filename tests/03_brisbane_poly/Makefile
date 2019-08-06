all: opt

vecadd.ll: vecadd.c
	clang -O3 -emit-llvm -S -o $@ $^

opt: vecadd.ll
	opt -load LLVMBrisbanePoly.so -brisbane-poly < vecadd.ll -o vecadd.bc
	llvm-dis vecadd.bc -o vecadd.bc.ll

clean:
	rm -f vecadd.ll vecadd.bc vecadd.bc.ll

