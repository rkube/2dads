include ../../Makefile_osx.inc

.PHONY: clean

test_derivs_spectral_host: test_derivs_spectral.cpp 
	$(CC) $(CFLAGS) $(INCLUDES) -DHOST test_derivs_spectral.cpp $(OBJ_DIR)/slab_bc_host.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/output.o -o test_derivs_spectral_host  $(LFLAGS) 

test_derivs_spectral_device: test_derivs_spectral.cu 
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -DDEVICE -o test_derivs_spectral_device $(OBJ_DIR)/slab_bc_device.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/output.o test_derivs_spectral.cu $(CUDALFLAGS) 
