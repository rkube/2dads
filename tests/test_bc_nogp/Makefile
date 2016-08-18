include ../../Makefile.inc

all: test_dtype test_dft
.PHONY: clean

test_dtype_host: test_dtype.cu 
	$(CC) $(CFLAGS) -DHOST -o test_dtype_host test_dtype.cpp $(INCLUDES) $(LFLAGS) 

test_dtype_device: test_dtype.cu 
	$(CUDACC) $(CUDACFLAGS) -DDEVICE -o test_dtype_device test_dtype.cu $(INCLUDES) $(CUDALFLAGS) -lcublas -lcusparse

test_slab_host: test_slab.cpp slab_bc.cpp slab_bc.h derivatives.h 
	$(CC) $(CFLAGS) -DHOST -o test_slab_host test_slab.cpp slab_bc.cpp $(INCLUDES) $(LFLAGS) -lfftw3 -lm

test_slab_device: test_slab.cu slab_bc.cu slab_bc.h derivatives.h 
	$(CUDACC) $(CUDACFLAGS) -DDEVICE -o test_slab_device test_slab.cu slab_bc.cu $(INCLUDES) $(CUDALFLAGS) -lfftw3 -lm 

test_derivs: slab_bc.cu slab_bc.h test_derivs.cu derivatives.h
	$(CUDACC) $(CUDACFLAGS) -DHOST -o test_derivs test_derivs.cu slab_bc.cu $(INCLUDES) $(CUDALFLAGS) -lcublas -lcusparse -lfftw3 -lm

test_cusolve: test_cusolve.cu derivatives.h
	$(CUDACC) $(CUDACFLAGS) -DHOST -o test_cusolve test_cusolve.cu $(INCLUDES) $(CUDALFLAGS) -lcublas -lcusparse -lfftw3 -lm

test_laplace: slab_bc.cu slab_bc.h derivatives.h test_laplace.cu
	$(CUDACC) $(CUDACFLAGS) -DHOST -o test_laplace test_laplace.cu slab_bc.cu $(INCLUDES) $(CUDALFLAGS) -lcublas -lcusparse -lfftw3 -lm

test_arakawa: slab_bc.cu slab_bc.h derivatives.h test_arakawa.cu
	$(CUDACC) $(CUDACFLAGS) -DHOST -o test_arakawa test_arakawa.cu slab_bc.cu $(INCLUDES) $(CUDALFLAGS) -lcublas -lcusparse -lfftw3 -lm

test_stiff: slab_bc.cu slab_bc.h test_stiff.cu
	$(CUDACC) $(CUDACFLAGS) -DHOST -o test_stiff test_stiff.cu slab_bc.cu $(INCLUDES) $(CUDALFLAGS) -lcublas -lcusparse -lfftw3 -lm

clean:
	rm test_dtype test_dft
