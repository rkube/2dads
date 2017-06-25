include ../../Makefile_osx.inc

.PHONY: clean

test_derivs_host: test_derivs.cpp 
	$(CC) $(CFLAGS) $(INCLUDES) -DHOST test_derivs.cpp $(OBJ_DIR)/slab_config_host.o -o test_derivs_host  $(LFLAGS) 

test_derivs_device: test_derivs.cpp
	$(CUDACC) $(CUDACFLAGS) $(INCLUDES) -DDEVICE -x cuda test_derivs.cpp -c -o test_derivs_device.o  
	$(CUDACC) test_derivs_device.o $(OBJ_DIR)/slab_config_device.o -o test_derivs_device $(CUDALFLAGS)
