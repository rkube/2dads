include ../../Makefile.inc

.PHONY: clean

test_dtype_host: test_dtype.cpp
	$(CC) $(CFLAGS) -DHOST -o test_dtype_host test_dtype.cpp $(INCLUDES) $(LFLAGS) 

test_dtype_device: test_dtype.cu 
	$(CUDACC) $(CUDACFLAGS) -DDEVICE -o test_dtype_device test_dtype.cu $(INCLUDES) $(CUDALFLAGS) 

test_slab_host: test_slab.cpp slab_bc.cpp slab_bc.h derivatives.h 
	$(CC) $(CFLAGS) -DHOST -o test_slab_host test_slab.cpp slab_bc.cpp $(INCLUDES) $(LFLAGS) 

test_slab_device: test_slab.cu slab_bc.cu slab_bc.h derivatives.h 
	$(CUDACC) $(CUDACFLAGS) -DDEVICE -o test_slab_device test_slab.cu slab_bc.cu $(INCLUDES) $(CUDALFLAGS) 

test_derivs_host: test_derivs.cpp slab_bc.cpp slab_bc.h derivatives.h 
	$(CC) $(CFLAGS) -DHOST -o test_derivs_host test_derivs.cpp slab_bc.cpp $(INCLUDES) $(LFLAGS) 

test_derivs_device: test_derivs.cu slab_bc.cu slab_bc.h derivatives.h 
	$(CUDACC) $(CUDACFLAGS) -DDEVICE -o test_derivs_device test_derivs.cu slab_bc.cu $(INCLUDES) $(CUDALFLAGS) 

test_arakawa_host: test_arakawa.cpp slab_bc.cpp slab_bc.h derivatives.h 
	$(CC) $(CFLAGS) -DHOST -o test_arakawa_host test_arakawa.cpp slab_bc.cpp $(INCLUDES) $(LFLAGS) 

test_arakawa_device: slab_bc.cu slab_bc.h derivatives.h test_arakawa.cu
	$(CUDACC) $(CUDACFLAGS) -DDEVICE -o test_arakawa_device test_arakawa.cu slab_bc.cu $(INCLUDES) $(CUDALFLAGS) 

test_laplace_host: test_laplace.cpp slab_bc.cpp slab_bc.h derivatives.h 
	$(CC) $(CFLAGS) -DHOST -o test_laplace_host test_laplace.cpp slab_bc.cpp $(INCLUDES) $(LFLAGS)

test_laplace_device: test_laplace.cu slab_bc.cu slab_bc.h derivatives.h 
	$(CUDACC) $(CUDACFLAGS) -DDEVICE -o test_laplace_device test_laplace.cu slab_bc.cu $(INCLUDES) $(CUDALFLAGS) 

#test_stiff: slab_bc.cu slab_bc.h test_stiff.cu
#	$(CUDACC) $(CUDACFLAGS) -DHOST -o test_stiff test_stiff.cu slab_bc.cu $(INCLUDES) $(CUDALFLAGS) -lcublas -lcusparse -lfftw3 -lm

clean:
	rm test_dtype_host test_dtype_device test_slab_host test_slab_device test_derivs_host test_derivs_device test_arakawa_host test_arakawa_device test_laplace_host test_laplace_device
