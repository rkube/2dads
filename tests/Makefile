include ../Makefile.inc

# .PHONY: 

# all: test_cucmplx test_array3 test_copy test_slab test_derivs test_stiff test_memcpy test_diff test_blob test_hw test_opengl


# Test the CuCmplx class, especially arithmetic and return types of the operators
test_cucmplx: test_cucmplx.cpp
	$(CC) $(CFLAGS) -o test_cucmplx test_cucmplx.cpp $(INCLUDES) 

# Test instantiation of cuda_array3: instantiation of real and complex type, arithmetic and output operators
test_array: test_array4.cu test_copy.cu
	# Test of array arithmetic
	#$(CUDACC) $(CUDACFLAGS) -DDEBUG -o test_array3 test_array3.cu $(INCLUDES) $(CUDALFLAGS)
	$(CUDACC) $(CUDACFLAGS) -DDEBUG -o test_array4 test_array4.cu $(INCLUDES) $(CUDALFLAGS)
	# Test of copy, move, advance operators
	$(CUDACC) $(CUDACFLAGS) -DDEBUG -o test_copy test_copy.cu $(INCLUDES) $(CUDALFLAGS)


# Test member functions of diag_array
# Compile a program that instantiates a cuda_array with NVCC first
test_diagarray: test_diagarray.cpp test_diagarray2.cpp ../obj/instant_cuda.o
	$(CUDACC) $(CUDACFLAGS) -c -o ../obj/instant_cuda.o  instant_cuda.cu $(INCLUDES) 
	$(CC) $(CFLAGS) -o test_diagarray test_diagarray.cpp ../obj/instant_cuda.o $(INCLUDES) $(LFLAGS)
	$(CC) $(CFLAGS) -o test_diagarray2 test_diagarray2.cpp ../obj/instant_cuda.o $(INCLUDES) $(LFLAGS)


test_cudadarray: test_cudadarray.cu ../include/cuda_darray.h
	$(CUDACC) $(CUDACFLAGS) -o test_cudadarray test_cudadarray.cu $(INCLUDES) $(LFLAGS)

# Run basic tests on slab structure
# Test initialization routine: Initialize with routine specified in input file, check that output state is consistent among all routines
test_slab: test_enum.cpp test_init.cpp test_derivs_space.cpp  ../obj/initialize.o ../obj/slab_config.o ../obj/slab_cuda.o
# Test enumeration of derivation kernels d_dx, d_dy, inv_lapl and time integrator stiff_k
	$(CC) $(CFLAGS) -o test_slab/test_enum ../obj/initialize.o ../obj/slab_config.o ../obj/slab_cuda.o test_enum.cpp $(INCLUDES) $(LFLAGS)
# Initialization
	$(CC) $(CFLAGS) -o test_slab/test_init ../obj/initialize.o ../obj/slab_config.o ../obj/slab_cuda.o test_init.cpp $(INCLUDES) $(LFLAGS)
# Test Fourier transformation 
#	$(CC) $(CCFLAGS) -o test_slab/test_dft $(OBJ_DIR)/initialize.o test_dft.cu $(INCLUDES) $(CUDALFLAGS)
# Test that derivation routines work
	$(CC) $(CFLAGS) -o test_slab/test_derivs ../obj/initialize.o ../obj/slab_config.o ../obj/slab_cuda.o test_derivs_space.cpp $(INCLUDES) $(LFLAGS)

# Test time integrator, solve diffusion problem and show how modes are integrated
test_stiff: test_stiff.cpp test_stiff_transient.cpp test_diff.cpp
	$(CC) $(CFLAGS) -o test_stiff/test_stiff ../obj/slab_cuda.o ../obj/slab_config.o ../obj/initialize.o ../obj/output.o test_stiff.cpp $(INCLUDES) $(LFLAGS)
	$(CC) $(CFLAGS) -o test_stiff/test_stiff_transient ../obj/slab_cuda.o ../obj/slab_config.o ../obj/initialize.o ../obj/output.o test_stiff_transient.cpp $(INCLUDES) $(LFLAGS)
	$(CC) $(CFLAGS) -o test_stiff/test_diff ../obj/slab_cuda.o ../obj/slab_config.o ../obj/initialize.o ../obj/output.o test_diff.cpp $(INCLUDES) $(LFLAGS)


# Test HDF5 output routines
test_output: test_output.cpp
	$(CC) $(CFLAGS) -o test_output/test_output ../obj/slab_cuda.o ../obj/slab_config.o ../obj/initialize.o ../obj/output.o test_output.cpp $(INCLUDES) $(LFLAGS)

# Test diagnostics
test_diag: test_diag.cpp
	$(CC) $(CFLAGS) -o test_diag ../obj/slab_cuda.o ../obj/slab_config.o ../obj/initialize.o ../obj/output.o test_diag.cpp $(INCLUDES) $(LFLAGS)

# Toy around with memcpu
test_memcpy: test_memcpy.cu
	$(CUDACC) -arch=sm_30 -O0 -o test_memcpy test_memcpy.cu $(INCLUDES) 

# Performance tests
test_perf: test_arrayperf.cpp
	$(CC) $(CFLAGS) -o test_perf/test_perf_arraybase test_arrayperf.cpp -pthread $(INCLUDES) $(LFLAGS) 

# Test integration of ky=0 modes with different diffusion coefficient
test_hw: test_hw.cpp
	$(CC) $(CFLAGS) -o test_hw/test_hw ../obj/slab_cuda.o ../obj/slab_config.o ../obj/initialize.o ../obj/output.o ../obj/diagnostics.o test_hw.cpp $(INCLUDES) $(LFLAGS) -I/usr/local/cuda

test_zfky0: test_zfky0.cpp test_zfky0_int.cpp
	$(CC) $(CFLAGS) -o test_zf/test_zfky0 ../obj/slab_cuda.o ../obj/slab_config.o ../obj/initialize.o ../obj/output.o ../obj/diagnostics.o test_zfky0.cpp $(INCLUDES) $(LFLAGS) -I/usr/local/cuda
	$(CC) $(CFLAGS) -o test_zf/test_zfky0_int ../obj/slab_cuda.o ../obj/slab_config.o ../obj/initialize.o ../obj/output.o ../obj/diagnostics.o test_zfky0_int.cpp $(INCLUDES) $(LFLAGS) -I/usr/local/cuda

# Tests involving OpenGL output
test_opengl: test_diff_opengl.cpp
	#$(CC) $(CFLAGS) -o test_opengl/test_init_opengl ../obj/initialize.o ../obj/slab_config.o ../obj/slab_cuda.o ../obj/shader.o ../obj/output.o test_init_opengl.cpp $(INCLUDES) $(LFLAGS) $(LFLAGS_OGL)
	#$(CC) $(CFLAGS) -o test_opengl/test_diff_opengl ../obj/initialize.o ../obj/slab_config.o ../obj/slab_cuda.o ../obj/shader.o ../obj/output.o test_diff_opengl1.cpp $(INCLUDES) $(LFLAGS) $(LFLAGS_OGL)
	$(CC) $(CFLAGS) -o test_opengl/test_diff_opengl2 ../obj/initialize.o ../obj/slab_config.o ../obj/slab_cuda.o ../obj/shader.o ../obj/output.o test_diff_opengl2.cpp $(INCLUDES) $(LFLAGS) $(LFLAGS_OGL)


clean:
	rm test_cucmplx test_array3 test_copy test_diagarray test_diagarray2 test_cudadarray test_slab/test_enum test_slab/test_init test_slab/test_derivs test_stiff/test_stiff test_stiff/test_stiff_transient test_stiff/test_diff test_output/test_output test_diag test_memcpy test_perf test_hw test_zf/test_zfky0 test_zf/test_zfky0_int test_opengl/test_diff_opengl2


