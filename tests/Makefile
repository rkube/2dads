include ../Makefile.inc

.PHONY: test_array2 test_arrayc test_slab test_init test_dft

all: test_array2 test_arrayc test_slab test_init test_dft

test_array2: 
	$(CC) $(CFLAGS) -o test_array2 $(OBJ_DIR)/cuda_array2.o test_array2.cpp $(IFLAGS) $(LFLAGS) 

test_arrayc: 
	$(CC) $(CFLAGS) -o test_arrayc $(OBJ_DIR)/cuda_array2.o test_arrayc.cpp $(IFLAGS) $(LFLAGS)

test_slab:
	$(CC) $(CFLAGS) -o test_slab $(OBJ_DIR)/cuda_array2.o $(OBJ_DIR)/slab_cuda.o $(OBJ_DIR)/slab_config.o test_slab.cpp $(IFLAGS) $(LFLAGS)

test_init:
	$(CC) $(CFLAGS) -o test_init $(OBJ_DIR)/initialize.o $(OBJ_DIR)/cuda_array2.o test_init.cpp $(IFLAGS) $(LFLAGS)

test_dft:
	$(CC) $(CFLAGS) -o test_dft $(OBJ_DIR)/initialize.o $(OBJ_DIR)/cuda_array2.o test_dft.cpp $(IFLAGS) $(LFLAGS)



clean:
	rm test_array2 test_arrayc test_slab test_init
