include ../Makefile.inc

.PHONY: test_cucmplx test_array3 test_copy test_slab test_init test_dft test_arith test_derivs test_output test_stiff test_memcpy test_diff test_blob test_hw test_diagarray test_perf

all: test_cucmplx test_array3 test_copy test_slab test_init test_dft test_arith test_derivs test_stiff test_memcpy test_diff test_blob test_hw


# Test the CuCmplx class, especially arithmetic and return types of the operators
test_cucmplx:
	$(CC) $(CFLAGS) -o test_cucmplx test_cucmplx.cpp $(INCLUDES) 

# Test instantiation of cuda_array3: instantiation of real and complex type, arithmetic and output operators
test_array3:
	# Test of array arithmetic
	$(CUDACC) $(CUDACFLAGS) -DDEBUG -o test_array3 test_array3.cu $(INCLUDES) $(CUDALFLAGS)
	# Test of copy, move, advance operators
	$(CUDACC) $(CUDACFLAGS) -DDEBUG -o test_copy test_copy.cu $(INCLUDES) $(CUDALFLAGS)


# Test member functions of diag_array
# Compile a program that instantiates a cuda_array with NVCC first
test_diagarray:
	$(CUDACC) $(CUDACFLAGS) -c -o ../obj/instant_cuda.o  instant_cuda.cu $(INCLUDES) 
	$(CC) $(CFLAGS) -o test_diagarray test_diagarray.cpp ../obj/instant_cuda.o $(INCLUDES) $(LFLAGS)
	$(CC) $(CFLAGS) -o test_diagarray2 test_diagarray2.cpp ../obj/instant_cuda.o $(INCLUDES) $(LFLAGS)


# Run basic tests on slab structure
# Test initialization routine: Initialize with routine specified in input file, check that output state is consistent among all routines
test_slab:
	$(CC) $(CFLAGS) -o test_slab/test_init $(OBJ_DIR)/initialize.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/slab_cuda.o test_init.cpp $(INCLUDES) $(LFLAGS)
# Test enumeration of derivation kernels d_dx, d_dy, inv_lapl and time integrator stiff_k
	$(CC) $(CFLAGS) -o test_slab/test_enum $(OBJ_DIR)/initialize.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/slab_cuda.o test_enum.cpp $(INCLUDES) $(LFLAGS)
# Initialization
	$(CC) $(CFLAGS) -o test_slab/test_init $(OBJ_DIR)/initialize.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/slab_cuda.o test_init.cpp $(INCLUDES) $(LFLAGS)
# Test Fourier transformation 
#	$(CC) $(CCFLAGS) -o test_slab/test_dft $(OBJ_DIR)/initialize.o test_dft.cu $(INCLUDES) $(CUDALFLAGS)
# Test that derivation routines work
	$(CC) $(CFLAGS) -o test_slab/test_derivs $(OBJ_DIR)/initialize.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/slab_cuda.o test_derivs_space.cpp $(INCLUDES) $(LFLAGS)

# Test time integrator, solve diffusion problem and show how modes are integrated
test_stiff:
	$(CC) $(CFLAGS) -o test_stiff/test_stiff $(OBJ_DIR)/slab_cuda.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/initialize.o test_stiff.cpp $(INCLUDES) $(LFLAGS)
	$(CC) $(CFLAGS) -o test_stiff/test_diff $(OBJ_DIR)/slab_cuda.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/initialize.o test_diff.cpp $(INCLUDES) $(LFLAGS)


# Test HDF5 output routines
test_output:
	$(CC) $(CFLAGS) -o test_output/test_output $(OBJ_DIR)/slab_cuda.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/initialize.o $(OBJ_DIR)/output.o test_output.cpp $(INCLUDES) $(LFLAGS)

# Test diagnostics
test_diag:
	$(CC) $(CFLAGS) -o test_diag $(OBJ_DIR)/slab_cuda.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/initialize.o $(OBJ_DIR)/output.o test_diag.cpp $(INCLUDES) $(LFLAGS)

# Toy around with memcpu
test_memcpy:
	$(CUDACC) -arch=sm_30 -O0 -o test_memcpy test_memcpy.cu $(INCLUDES) 
# Performance tests
test_perf:
	$(CC) $(CFLAGS) -o test_perf/test_perf_arraybase test_arrayperf.cpp -pthread $(INCLUDES) $(LFLAGS) 

# Test integration of ky=0 modes with different diffusion coefficient
test_hw:
	$(CC) $(CFLAGS) -o test_hw/test_hw $(OBJ_DIR)/slab_cuda.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/initialize.o $(OBJ_DIR)/output.o $(OBJ_DIR)/diagnostics.o test_hw.cpp $(INCLUDES) $(LFLAGS) -I/usr/local/cuda

test_zfky0:
	$(CC) $(CFLAGS) -o test_zf/test_zfky0 $(OBJ_DIR)/slab_cuda.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/initialize.o $(OBJ_DIR)/output.o $(OBJ_DIR)/diagnostics.o test_zfky0.cpp $(INCLUDES) $(LFLAGS) -I/usr/local/cuda
	$(CC) $(CFLAGS) -o test_zf/test_zfky0_int $(OBJ_DIR)/slab_cuda.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/initialize.o $(OBJ_DIR)/output.o $(OBJ_DIR)/diagnostics.o test_zfky0_int.cpp $(INCLUDES) $(LFLAGS) -I/usr/local/cuda


clean:
	rm test_array2 test_arrayc test_slab test_init test_arith test_deriv test_diff


