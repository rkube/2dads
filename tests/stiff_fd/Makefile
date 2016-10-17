include ../../Makefile_linux.inc

.PHONY: clean

test_stiff_host: test_stiff.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -DHOST -o test_stiff_host $(OBJ_DIR)/slab_bc_host.o test_stiff.cpp $(LFLAGS) 

test_stiff_device: test_stiff.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -DDEVICE -o test_stiff_device $(OBJ_DIR)/slab_bc_device.o test_stiff.cu $(CUDALFLAGS) 
