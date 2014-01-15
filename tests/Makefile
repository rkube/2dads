include ../Makefile.inc

.PHONY: test_array2 test_arrayc test_copy test_slab test_init test_dft test_arith test_deriv test_output test_stiff test_memcpy test_diff test_blob test_hw test_diagarray

all: test_array2 test_arrayc test_copy test_slab test_init test_dft test_arith test_deriv test_stiff test_memcpy test_diff test_blob test_hw

test_array2: 
	$(CC) $(CFLAGS) -o test_array2 $(OBJ_DIR)/cuda_array2.o test_array2.cpp $(IFLAGS) $(LFLAGS) 

test_arith:
	$(CC) $(CFLAGS) -o test_arith $(OBJ_DIR)/cuda_array2.o test_arith.cpp $(IFLAGS) $(LFLAGS)

test_arrayc: 
	$(CC) $(CFLAGS) -o test_arrayc $(OBJ_DIR)/cuda_array2.o test_arrayc.cpp $(IFLAGS) $(LFLAGS)

test_copy:
	$(CC) $(CFLAGS) -o test_copy $(OBJ_DIR)/cuda_array2.o test_copy.cpp $(IFLAGS) $(LFLAGS)

test_slab:
	$(CC) $(CFLAGS) -o test_slab $(OBJ_DIR)/cuda_array2.o $(OBJ_DIR)/slab_cuda.o $(OBJ_DIR)/slab_cuda2.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/initialize.o $(OBJ_DIR)/output.o test_slab.cpp $(IFLAGS) $(LFLAGS)

test_init:
	$(CC) $(CFLAGS) -o test_init $(OBJ_DIR)/initialize.o  $(OBJ_DIR)/cuda_array2.o test_init.cpp $(IFLAGS) $(LFLAGS)

test_dft:
	$(CC) $(CFLAGS) -o test_dft $(OBJ_DIR)/initialize.o $(OBJ_DIR)/cuda_array2.o test_dft.cpp $(IFLAGS) $(LFLAGS)

test_deriv:
	$(CUDACC) -arch=sm_30 -O0 -o test_deriv $(OBJ_DIR)/initialize.o $(OBJ_DIR)/cuda_array2.o test_deriv.cu $(IFLAGS) $(LFLAGS)

test_memcpy:
	$(CUDACC) -arch=sm_30 -O0 -o test_memcpy test_memcpy.cu $(IFLAGS) 

test_output:
	$(CC) $(CFLAGS) -DDEBUG -o test_output $(OBJ_DIR)/cuda_array2.o $(OBJ_DIR)/slab_cuda.o $(OBJ_DIR)/slab_cuda2.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/initialize.o $(OBJ_DIR)/output.o test_output.cpp $(IFLAGS) $(LFLAGS)

test_stiff:
	$(CC) $(CFLAGS) -o test_stiff $(OBJ_DIR)/cuda_array2.o $(OBJ_DIR)/slab_cuda.o $(OBJ_DIR)/slab_cuda2.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/initialize.o $(OBJ_DIR)/output.o test_stiff.cpp $(IFLAGS) $(LFLAGS)

test_diff:
	$(CC) $(CFLAGS) -o test_diff/test_diff $(OBJ_DIR)/cuda_array2.o $(OBJ_DIR)/slab_cuda.o $(OBJ_DIR)/slab_cuda2.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/initialize.o $(OBJ_DIR)/output.o test_diff.cpp $(IFLAGS) $(LFLAGS)

test_blob:
	$(CC) $(CFLAGS) -o test_blob/test_blob $(OBJ_DIR)/cuda_array2.o $(OBJ_DIR)/slab_cuda.o $(OBJ_DIR)/slab_cuda2.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/initialize.o $(OBJ_DIR)/output.o $(OBJ_DIR)/diagnostics.o test_blob.cpp $(IFLAGS) $(LFLAGS)

test_hw:
	$(CC) $(CFLAGS) -o test_hw/test_hw $(OBJ_DIR)/cuda_array2.o $(OBJ_DIR)/slab_cuda.o $(OBJ_DIR)/slab_cuda2.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/initialize.o $(OBJ_DIR)/output.o $(OBJ_DIR)/diag_array.o $(OBJ_DIR)/array_base.o $(OBJ_DIR)/diagnostics.o test_hw.cpp $(IFLAGS) $(LFLAGS)

test_diagarray:
	$(CC) $(CFLAGS) -o test_diagarray $(OBJ_DIR)/cuda_array2.o $(OBJ_DIR)/array_base.o $(OBJ_DIR)/diag_array.o test_diagarray.cpp $(IFLAGS) $(LFLAGS)

clean:
	rm test_array2 test_arrayc test_slab test_init test_arith test_deriv test_diff
