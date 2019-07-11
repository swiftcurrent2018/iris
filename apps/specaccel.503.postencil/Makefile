BRISBANE=../../install
LDFLAGS=-L$(BRISBANE)/lib -L$(HOME)/install/lib -lbrisbane -lOpenCL -lpthread

all: postencil postencil-brisbane

postencil: file.c kernels.c main.c parboil.c specrand.c
	g++ -I . -DSPEC -o $@ $^

postencil-brisbane: file.c main-brisbane.c parboil.c specrand.c
	g++ -I. -I$(BRISBANE)/include -DSPEC -o $@ $^ $(LDFLAGS)

run: postencil
	./$^ -o 64x64x64.out 64 64 64 10

run-brisbane: postencil-brisbane
	./$^ -o 64x64x64.out-brsbane 64 64 64 10

clean:
	rm -f postencil postencil-brisbane
