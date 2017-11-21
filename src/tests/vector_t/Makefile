include ../../Makefile_linux.inc

test_vector_host: test_vector.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -DHOST -o test_vector_host test_vector.cpp $(LFLAGS)

test_vector_device: test_vector.cpp
	$(CUDACC) $(CUDACFLAGS) $(INCLUDES) -DDEVICE -x cuda -v -o test_vector_device test_vector.cpp $(CUDALFLAGS)

test_accumulate_host: test_accumulate.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -DHOST -o test_accumulate_host test_accumulate.cpp $(LFLAGS)


