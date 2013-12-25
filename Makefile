#CC	= /opt/local/bin/g++-mp-4.8
#CC	= /usr/bin/g++
#CFLAGS = -O0 -g 
#IFLAGS = -I/home/rku000/cuda-workspace/cuda_array2/ -I/usr/local/cuda/include
#LFLAGS = -L/usr/local/lib -L/usr/local/cuda/lib -lcudart -lcufft
#
#CUDACC	= /usr/local/cuda/bin/nvcc
#CUDACFLAGS	= -O0 -G --ptxas-options=-v --gpu-architecture sm_30
#CUDALFLAGS	= -lcufft
#
include Makefile.inc
# Subdirectories
TEST_DIR = tests


.PHONY: cuda_array2 slab_cuda clean tests

all: cuda_array2 slab_cuda tests

cuda_array2:
	$(CUDACC) $(CUDACFLAGS) -c -o obj/cuda_array2.o cuda_array2.cu $(IFLAGS)

initialize:
	$(CUDACC) $(CUDACFLAGS) -c -o obj/initialize.o initialize.cu $(IFLAGS)

slab_config:
	$(CC) $(CFLAGS) -c -o obj/slab_config.o slab_config.cpp $(IFLAGS)

slab_cuda: slab_config initialize
	$(CC) $(CFLAGS) -c -o obj/slab_cuda.o obj/initialize.o slab_cuda.cpp $(IFLAGS)


tests: cuda_array2 slab_cuda
	$(MAKE) -C $(TEST_DIR)

clean:
	rm obj/*.o


