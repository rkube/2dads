include ../../Makefile_osx.inc

.PHONY: clean

test_arakawa_host: test_arakawa.cpp 
	$(CC) $(CFLAGS) $(INCLUDES) -DHOST $(OBJ_DIR)/slab_config.o -o test_arakawa_host  test_arakawa.cpp $(LFLAGS) 

test_arakawa_device: test_arakawa.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -DDEVICE -o test_arakawa_device $(OBJ_DIR)/slab_bc_device.o test_arakawa.cu $(CUDALFLAGS) 
