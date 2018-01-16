include ../../Makefile_linux.inc

.PHONY: clean

test_laplace_host: test_laplace.cpp 
	$(CC) $(CFLAGS) $(INCLUDES) -DHOST -o test_laplace_host $(OBJ_DIR)/slab_bc_host.o $(OBJ_DIR)/output.o $(OBJ_DIR)/diagnostics_host.o $(OBJ_DIR)/slab_config.o test_laplace.cpp $(LFLAGS)

test_laplace_device: test_laplace.cu 
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -DDEVICE -o test_laplace_device $(OBJ_DIR)/slab_bc_device.o $(OBJ_DIR)/output.o $(OBJ_DIR)/slab_config.o test_laplace.cu $(CUDALFLAGS) 
