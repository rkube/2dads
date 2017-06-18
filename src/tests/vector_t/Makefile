include ../../Makefile_osx.inc

test_vector_host: test_vector.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -DHOST -o test_vector_host test_vector.cpp $(LFLAGS)

test_vector_device: test_vector.cpp
	$(CC) $(CUDACFLAGS) $(INCLUDES) -DDEVICE -x cuda -o test_vector_device test_vector.cpp $(CUDALFLAGS)
