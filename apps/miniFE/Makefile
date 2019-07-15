BRISBANE=../../install

MINIFE_TYPES=-DMINIFE_SCALAR=double -DMINIFE_LOCAL_ORDINAL=int -DMINIFE_GLOBAL_ORDINAL=int
MINIFE_MATRIX_TYPE=-DMINIFE_CSR_MATRIX

CXX=g++
CFLAGS=-g -I. -I$(BRISBANE)/include $(MINIFE_TYPES) $(MINIFE_MATRIX_TYPE)
LDFLAGS=-L$(BRISBANE)/lib -L$(HOME)/install/lib -lbrisbane -lOpenCL -pthread

all: miniFE miniFE-brisbane

miniFE: main.cpp BoxPartition.cpp mytimer.cpp YAML_Doc.cpp YAML_Element.cpp param_utils.cpp utils.cpp
	$(CXX) $(CFLAGS) -fopenmp -o $@ $^

miniFE-brisbane: main-brisbane.cpp BoxPartition.cpp mytimer.cpp YAML_Doc.cpp YAML_Element.cpp param_utils.cpp utils.cpp
	$(CXX) $(CFLAGS) -fopenmp -o $@ $^ $(LDFLAGS)

generate_info:
	./generate_info_header "$(CXX)" "$(CFLAGS)" "miniFE" "MINIFE"

clean:
	rm -f miniFE miniFE-brisbane miniFE.*.yaml
