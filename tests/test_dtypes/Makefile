include ../../Makefile.inc

SOURCES_CU = test_memcpy.cu
SOURCES_CPP = test_cucmplx.cpp

all: test_memcpy test_cucmplx
.PHONY: clean

test_memcpy: test_memcpy.cu
	$(CUDACC) $(CUDACFLAGS) -DDEBUG -o test_memcpy test_memcpy.cu $(INCLUDES) $(CUDALFLAGS)

test_cucmplx: test_cucmplx.cpp $(PROJ_INCDIR)/cucmplx.h
	$(CC) $(CFLAGS) -o test_cucmplx test_cucmplx.cpp $(INCLUDES)

test_operators: test_operators.cu $(PROJ_INCDIR)/cuda_operators.h
	$(CUDACC) $(CUDACFLAGS) -DDEBUG -o test_operators test_operators.cu $(INCLUDES) $(CUDALFLAGS)

clean:
	rm test_memcpy test_cucmplx
