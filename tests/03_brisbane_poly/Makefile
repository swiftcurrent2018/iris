all:hello.ll hello

hello.ll:hello.c
	clang -O3 -emit-llvm -S -o $@ $^

hello:hello.c
	clang -O3 -o $@ $^

clean:
	rm -f hello.ll hello

