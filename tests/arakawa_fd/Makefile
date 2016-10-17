include ../../Makefile_linux.inc

.PHONY: clean

test_arakawa_host: test_arakawa.cpp 
	$(CC) $(CFLAGS) $(INCLUDES) -DHOST -o test_arakawa_host $(OBJ_DIR)/slab_bc_host.o test_arakawa.cpp slab_bc.cpp $(LFLAGS) 

test_arakawa_device: test_arakawa.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -DDEVICE -o test_arakawa_device $(OBJ_DIR)/slab_bc_host.o test_arakawa.cu $(CUDALFLAGS) 
