include ../../Makefile_linux.inc

.PHONY: clean

test_dtype_host: test_dtype.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -DHOST -o test_dtype_host test_dtype.cpp $(LFLAGS) 

test_dtype_device: test_dtype.cu 
	$(CUDACC) $(CUDACFLAGS) $(INCLUDES) -DDEVICE -o test_dtype_device test_dtype.cu $(CUDALFLAGS)
	#$(CUDACC) -std=c++14 -stdlib=libc++ --cuda-gpu-arch=sm_50 -I/home/rku000/source/2dads/include -o test_dtype_device test_dtype.cu   
	#-lcudart_static -ldl -lm -lpthread -lrt

clean:
	rm test_dtype_host test_dtype_device 
