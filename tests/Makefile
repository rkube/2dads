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

test_slab:
	$(CC) $(CFLAGS) -o test_slab $(OBJ_DIR)/slab_cuda.o $(OBJ_DIR)/slab_cuda2.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/initialize.o $(OBJ_DIR)/output.o test_slab.cpp $(IFLAGS) $(LFLAGS)

test_init:
	$(CC) $(CFLAGS) -o test_init $(OBJ_DIR)/initialize.o  $(OBJ_DIR)/cuda_array2.o test_init.cpp $(IFLAGS) $(LFLAGS)

test_dft:
	$(CUDACC) $(CUDACFLAGS) -o test_dft $(OBJ_DIR)/initialize.o test_dft.cu $(IFLAGS) $(CUDALFLAGS)

test_derivs:
	$(CC) $(CFLAGS) -o test_derivs/test_derivs $(OBJ_DIR)/cuda_array2.o $(OBJ_DIR)/slab_cuda.o $(OBJ_DIR)/slab_cuda2.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/initialize.o $(OBJ_DIR)/output.o test_derivs.cpp $(IFLAGS) $(LFLAGS)

test_memcpy:
	$(CUDACC) -arch=sm_30 -O0 -o test_memcpy test_memcpy.cu $(IFLAGS) 

test_output:
	$(CC) $(CFLAGS) -DDEBUG -o test_output $(OBJ_DIR)/cuda_array2.o $(OBJ_DIR)/slab_cuda.o $(OBJ_DIR)/slab_cuda2.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/initialize.o $(OBJ_DIR)/output.o test_output.cpp $(IFLAGS) $(LFLAGS)

test_stiff:
	$(CC) $(CFLAGS) -DDEBUG -o test_stiff/test_stiff $(OBJ_DIR)/cuda_array2.o $(OBJ_DIR)/slab_cuda.o $(OBJ_DIR)/slab_cuda2.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/initialize.o $(OBJ_DIR)/output.o test_stiff.cpp $(IFLAGS) $(LFLAGS)

test_diff:
	$(CC) $(CFLAGS) -o test_diff/test_diff $(OBJ_DIR)/cuda_array2.o $(OBJ_DIR)/slab_cuda.o $(OBJ_DIR)/slab_cuda2.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/initialize.o $(OBJ_DIR)/output.o test_diff.cpp $(IFLAGS) $(LFLAGS)

test_blob:
	$(CC) $(CFLAGS) -o test_blob/test_blob $(OBJ_DIR)/cuda_array2.o $(OBJ_DIR)/slab_cuda.o $(OBJ_DIR)/slab_cuda2.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/initialize.o $(OBJ_DIR)/output.o $(OBJ_DIR)/diagnostics.o test_blob.cpp $(IFLAGS) $(LFLAGS)

test_hw:
	$(CC) $(CFLAGS) -o test_hw/test_hw $(OBJ_DIR)/cuda_array2.o $(OBJ_DIR)/slab_cuda.o $(OBJ_DIR)/slab_cuda2.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/initialize.o $(OBJ_DIR)/output.o $(OBJ_DIR)/diagnostics.o test_hw.cpp $(IFLAGS) $(LFLAGS)

test_diagarray:
	$(CC) $(CFLAGS) -o test_diagarray $(OBJ_DIR)/cuda_array2.o test_diagarray.cpp $(IFLAGS) $(LFLAGS)

clean:
	rm test_array2 test_arrayc test_slab test_init test_arith test_deriv test_diff



#test_array2: 
#	$(CC) $(CFLAGS) -o test_array2 $(OBJ_DIR)/cuda_array2.o test_array2.cpp $(IFLAGS) $(LFLAGS) 



