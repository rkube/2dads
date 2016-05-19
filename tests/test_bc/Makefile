include ../../Makefile.inc

SOURCES_CU = test_memcpy.cu
SOURCES_CPP = test_cucmplx.cpp

all: test_bc test_dft test_slab
.PHONY: clean

test_bc: test_bc.cu 
	$(CUDACC) $(CUDACFLAGS) -o test_bc test_bc.cu $(INCLUDES) $(CUDALFLAGS)

test_dft: test_dft.cu 
	$(CUDACC) $(CUDACFLAGS) -o test_dft test_dft.cu $(INCLUDES) $(CUDALFLAGS)

test_slab: test_slab.cu 
	$(CUDACC) $(CUDACFLAGS) -DDEBUG -o test_slab test_slab.cu $(INCLUDES) $(CUDALFLAGS)

clean:
	rm test_memcpy test_cucmplx
