include ../../Makefile.inc

all: test_dtype test_dft
.PHONY: clean

test_dtype: test_dtype.cu 
	$(CUDACC) $(CUDACFLAGS) -o test_dtype test_dtype.cu $(INCLUDES) $(CUDALFLAGS)

test_dft: test_dft.cu 
	$(CUDACC) $(CUDACFLAGS) -o test_dft test_dft.cu $(INCLUDES) $(CUDALFLAGS)

test_slab: slab_bc.cu slab_bc.h test_slab.cu
	$(CUDACC) $(CUDACFLAGS) -o test_slab test_slab.cu slab_bc.cu $(INCLUDES) $(CUDALFLAGS)

test_derivs: slab_bc.cu slab_bc.h test_derivs.cu derivatives.h
	$(CUDACC) $(CUDACFLAGS) -o test_derivs test_derivs.cu slab_bc.cu $(INCLUDES) $(CUDALFLAGS)

clean:
	rm test_dtype test_dft
