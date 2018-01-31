include ../../Makefile_linux.inc

.PHONY: clean

test_derivs_host: test_derivs.cpp 
	$(CC) $(CFLAGS) $(INCLUDES) -DHOST test_derivs.cpp $(OBJ_DIR)/slab_bc_host.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/output.o $(OBJ_DIR)/diagnostics_host.o -o test_derivs_host  $(LFLAGS) 

test_derivs_device: test_derivs.cu 
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -DDEVICE -o test_derivs_device $(OBJ_DIR)/slab_bc_device.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/output.o test_derivs.cu $(CUDALFLAGS) 
