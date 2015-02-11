include ../../Makefile.inc

SOURCES_CU = test_array3.cu test_array4.cu test_copy.cu test_cudadarray.cu test_enum.cu
SOURCES_CPP = test_diagarray2.cpp test_diagarray.cpp

all: test_array4 test_enum test_array_so test_cudadarray test_diagarray

# Test constructors
test_constructors: test_constructors.cu $(PROJ_INCDIR)/cuda_array4.h
	$(CUDACC) $(CUDACFLAGS) -DDEBUG -o test_constructors test_constructors.cu $(INCLUDES) $(CUDALFLAGS)

# Test of array arithmetic
test_array4: test_array4.cu $(PROJ_INCDIR)/cuda_array4.h 
	$(CUDACC) $(CUDACFLAGS) -DDEBUG -o test_array4 test_array4.cu $(INCLUDES) $(CUDALFLAGS)

# Test of enumeration
test_enum: test_enum.cu $(PROJ_INCDIR)/cuda_array4.h
	$(CUDACC) $(CUDACFLAGS) -o test_enum test_enum.cu $(INCLUDES) $(CUDALFLAGS)

# provide ctypes interface to arithmetic operators
test_array_so: test_array4_py.cu $(PROJ_INCDIR)/cuda_array4.h
	$(CUDACC) $(CUDACFLAGS) -DDEBUG -shared -Xcompiler -fPIC -Xlinker -soname=test_array4 -o test_array4.so test_array4_py.cu $(INCLUDES) $(CUDALFLAGS)

# Test of copy, move, advance operators
test_copy: test_copy.cu $(PROJ_INCDIR)/cuda_array4.h 
	$(CUDACC) $(CUDACFLAGS) -DDEBUG -o test_copy test_copy.cu $(INCLUDES) $(CUDALFLAGS)

# test cuda diagnostic array 
test_cudadarray: test_cudadarray.cu $(PROJ_INCDIR)/cuda_darray.h  $(PROJ_INCDIR)/cuda_array4.h 
	$(CUDACC) $(CUDACFLAGS) -o test_cudadarray test_cudadarray.cu $(INCLUDES) $(LFLAGS)

# cuda diagnostic array arithmetic and operator return types
test_cudadarray_arith: test_cudadarray_arith.cu $(PROJ_INCDIR)/cuda_darray.h $(PROJ_INCDIR)/cuda_array4.h  
	$(CUDACC) $(CUDACFLAGS) -o test_cudadarray_arith test_cudadarray_arith.cu $(INCLUDES) $(LFLAGS)

# test upcast methods in cuda_darray
test_upcast: test_upcast.cu $(PROJ_INCDIR)/cuda_darray.h $(PROJ_INCDIR)/cuda_array4.h
	$(CUDACC) $(CUDACFLAGS) -o test_upcast test_upcast.cu $(INCLUDES) $(LFLAGS)

#test diagnostic arrays
test_diagarray: test_diagarray.cpp test_diagarray2.cpp $(PROJ_INCDIR)/diag_array.h
	$(CC) $(CFLAGS) -o test_diagarray test_diagarray.cpp ../obj/instant_cuda.o $(INCLUDES) $(LFLAGS)
	$(CC) $(CFLAGS) -o test_diagarray2 test_diagarray2.cpp ../obj/instant_cuda.o $(INCLUDES) $(LFLAGS)

clean:
	rm test_array4 test_enum test_array4.so test_copy test_cudadarray test_diagarray test_diagarray2


