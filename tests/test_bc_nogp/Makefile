include ../../Makefile_linux.inc

.PHONY: clean

test_dtype_host: test_dtype.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -DHOST -o test_dtype_host test_dtype.cpp $(LFLAGS) 

test_dtype_device: test_dtype.cu 
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -DDEVICE -o test_dtype_device test_dtype.cu $(CUDALFLAGS) 

clean:
	rm test_dtype_host test_dtype_device 
