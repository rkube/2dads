include ../Makefile.inc

.PHONY: test_cucmplx test_array3 test_copy test_slab test_init test_dft test_arith test_derivs test_output test_stiff test_memcpy test_diff test_blob test_hw test_diagarray

all: test_cucmplx test_array3 test_copy test_slab test_init test_dft test_arith test_derivs test_stiff test_memcpy test_diff test_blob test_hw


# Test the CuCmplx class, especially arithmetic and return types of the operators
test_cucmplx:
	$(CC) $(CFLAGS) -o test_cucmplx test_cucmplx.cpp $(IFLAGS) 

# Test instantiation of cuda_array3: instantiation of real and complex type, arithmetic and output operators
test_array3:
	$(CUDACC) $(CUDACFLAGS) -o test_array3 test_array3.cu $(IFLAGS) $(CUDALFLAGS)

# Test copy, move and advance operators of cuda_array3
test_copy:
	$(CC) $(CFLAGS) -o test_copy $(OBJ_DIR)/cuda_array2.o test_copy.cpp $(IFLAGS) $(LFLAGS)

# Run basic tests on slab structure
# Test initialization routine: Initialize with routine specified in input file, check that output state is consistent among all routines
test_slab:
	$(CC) $(CFLAGS) -o test_slab/test_init $(OBJ_DIR)/initialize.o $(OBJ_DIR)/slab_cuda.o $(OBJ_DIR)/slab_config.o test_init.cpp $(IFLAGS) $(LFLAGS)
# Test enumeration of derivation kernels d_dx, d_dy, inv_lapl and time integrator stiff_k
	$(CC) $(CFLAGS) -o test_slab/test_enum $(OBJ_DIR)/slab_cuda.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/initialize.o test_enum.cpp $(IFLAGS) $(LFLAGS)


# Test that Fourier transformation work 
test_dft:
	$(CUDACC) $(CUDACFLAGS) -o test_slab/test_dft $(OBJ_DIR)/initialize.o test_dft.cu $(IFLAGS) $(CUDALFLAGS)

# Test that derivation routines work
test_derivs:
	$(CC) $(CFLAGS) -o test_slab/test_derivs $(OBJ_DIR)/slab_cuda.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/initialize.o test_derivs.cpp $(IFLAGS) $(LFLAGS)

# Test time integrator, solve diffusion problem and show how modes are integrated
test_stiff:
	$(CC) $(CFLAGS) -o test_stiff/test_stiff $(OBJ_DIR)/slab_cuda.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/initialize.o test_stiff.cpp $(IFLAGS) $(LFLAGS)
	$(CC) $(CFLAGS) -o test_stiff/test_diff $(OBJ_DIR)/slab_cuda.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/initialize.o test_diff.cpp $(IFLAGS) $(LFLAGS)


# Test HDF5 output routines
test_output:
	$(CC) $(CFLAGS) -o test_output/test_output $(OBJ_DIR)/slab_cuda.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/initialize.o $(OBJ_DIR)/output.o test_output.cpp $(IFLAGS) $(LFLAGS)


# Test diagnostics
test_diag:
	$(CC) $(CFLAGS) -o test_diag $(OBJ_DIR)/slab_cuda.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/initialize.o $(OBJ_DIR)/output.o test_diag.cpp $(IFLAGS) $(LFLAGS)

# Toy around with memcpu
test_memcpy:
	$(CUDACC) -arch=sm_30 -O0 -o test_memcpy test_memcpy.cu $(IFLAGS) 

test_blob:
	$(CC) $(CFLAGS) -o test_blob/test_blob $(OBJ_DIR)/cuda_array2.o $(OBJ_DIR)/slab_cuda.o $(OBJ_DIR)/slab_cuda2.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/initialize.o $(OBJ_DIR)/output.o $(OBJ_DIR)/diagnostics.o test_blob.cpp $(IFLAGS) $(LFLAGS)

test_hw:
	$(CC) $(CFLAGS) -o test_hw/test_hw $(OBJ_DIR)/slab_cuda.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/initialize.o $(OBJ_DIR)/output.o $(OBJ_DIR)/diagnostics.o test_hw.cpp $(IFLAGS) $(LFLAGS)


clean:
	rm test_array2 test_arrayc test_slab test_init test_arith test_deriv test_diff


