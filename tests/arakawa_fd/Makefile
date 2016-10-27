include ../../Makefile_linux.inc

.PHONY: clean

test_arakawa_host: test_arakawa.cpp 
	$(CC) $(CFLAGS) $(INCLUDES) -DHOST -o test_arakawa_host $(OBJ_DIR)/slab_bc_host.o $(OBJ_DIR)/output.o $(OBJ_DIR)/slab_config.o test_arakawa.cpp $(LFLAGS) 

test_arakawa_device: test_arakawa.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -DDEVICE -o test_arakawa_device $(OBJ_DIR)/slab_bc_device.o test_arakawa.cu $(CUDALFLAGS) 
