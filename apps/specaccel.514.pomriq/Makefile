BRISBANE=../../install

all: pomirq pomirq-brisbane

pomirq: file.c main.c parboil.c
	g++ -I . -DSPEC -o $@ $^

pomirq-brisbane: file.c main-brisbane.c parboil.c
	g++ -I. -I$(BRISBANE)/include -DSPEC -o $@ $^ -L$(BRISBANE)/lib -lbrisbane -lOpenCL -lpthread

run: pomirq
	./$^ -i 32_32_32_dataset.bin -o 32_32_32_dataset.out

run-brisbane: pomirq-brisbane
	./$^ -i 32_32_32_dataset.bin -o 32_32_32_dataset.out-brisbane

clean:
	rm -f pomirq pomirq-brisbane
