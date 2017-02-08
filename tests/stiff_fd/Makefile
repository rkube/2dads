include ../../Makefile_linux.inc

.PHONY: clean

test_diff_host: test_diff.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -DHOST -o test_diff_host $(OBJ_DIR)/slab_bc_host.o $(OBJ_DIR)/output.o $(OBJ_DIR)/slab_config.o test_diff.cpp $(LFLAGS) 

test_stiff_host: test_stiff.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -DHOST -o test_stiff_host $(OBJ_DIR)/slab_bc_host.o $(OBJ_DIR)/output.o $(OBJ_DIR)/slab_config.o test_stiff.cpp $(LFLAGS) 

test_stiff_device: test_stiff.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -DDEVICE -o test_stiff_device $(OBJ_DIR)/slab_bc_device.o $(OBJ_DIR)/output.o $(OBJ_DIR)/slab_config.o test_stiff.cu $(CUDALFLAGS) 
