include ../../Makefile_osx.inc

.PHONY: clean

test_dtype_host: test_dtype.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -DHOST -o test_dtype_host test_dtype.cpp $(LFLAGS) 

test_dtype_device: test_dtype.cu 
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -DDEVICE -o test_dtype_device test_dtype.cu $(CUDALFLAGS) 

test_derivs_host: test_derivs.cpp 
	$(CC) $(CFLAGS) $(INCLUDES) -DHOST test_derivs.cpp $(OBJ_DIR)/slab_bc_host.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/output.o -o test_derivs_host  $(LFLAGS) 

test_derivs_device: test_derivs.cu 
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -DDEVICE -o test_derivs_device $(OBJ_DIR)/slab_bc_device.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/output.o test_derivs.cu $(CUDALFLAGS) 

test_derivs_spectral_host: test_derivs_spectral.cpp 
	$(CC) $(CFLAGS) $(INCLUDES) -DHOST test_derivs_spectral.cpp $(OBJ_DIR)/slab_bc_host.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/output.o -o test_derivs_spectral_host  $(LFLAGS) 

test_derivs_spectral_device: test_derivs_spectral.cu 
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -DDEVICE -o test_derivs_spectral_device $(OBJ_DIR)/slab_bc_device.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/output.o test_derivs_spectral.cu $(CUDALFLAGS) 

test_arakawa_host: test_arakawa.cpp 
	$(CC) $(CFLAGS) $(INCLUDES) -DHOST -o test_arakawa_host $(OBJ_DIR)/slab_bc_host.o test_arakawa.cpp slab_bc.cpp $(LFLAGS) 

test_arakawa_device: test_arakawa.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -DDEVICE -o test_arakawa_device $(OBJ_DIR)/slab_bc_host.o test_arakawa.cu $(CUDALFLAGS) 

test_laplace_host: test_laplace.cpp 
	$(CC) $(CFLAGS) $(INCLUDES) -DHOST -o test_laplace_host $(OBJ_DIR)/slab_bc_host.o test_laplace.cpp $(LFLAGS)

test_laplace_device: test_laplace.cu 
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -DDEVICE -o test_laplace_device $(OBJ_DIR)/slab_bc_device.o test_laplace.cu $(CUDALFLAGS) 

test_stiff_host: test_stiff.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -DHOST -o test_stiff_host $(OBJ_DIR)/slab_bc_host.o test_stiff.cpp $(LFLAGS) 

test_stiff_device: test_stiff.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -DDEVICE -o test_stiff_device $(OBJ_DIR)/slab_bc_device.o test_stiff.cu $(CUDALFLAGS) 

clean:
	rm test_dtype_host test_dtype_device test_slab_host test_slab_device test_derivs_host test_derivs_device test_arakawa_host test_arakawa_device test_laplace_host test_laplace_device
