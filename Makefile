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

.PHONY: slab_cuda clean tests

all: cuda_array2 slab_cuda tests

#common_kernel:
#	$(CUDACC) $(CUDACFLAGS) -o obj/common_kernel.o common_kernel.cu $(IFLAGS)

cuda_array2: 
	$(CUDACC) $(CUDACFLAGS) -o obj/cuda_array2.o cuda_array2.cu $(IFLAGS)
	#$(CUDACC) $(CUDACFLAGS) -dc common_kernel.cu cuda_array2.cu $(IFLAGS)
	#$(CUDACC) -arch=sm_30 -dlink common_kernel.o cuda_array2.o -o obj/cuda_array2_linked.o
	#mv common_kernel.o obj/common_kernel_inc.o
	#mv cuda_array2.o obj/cuda_array2_inc.o

initialize: 
	$(CUDACC) $(CUDACFLAGS) -o obj/initialize.o initialize.cu $(IFLAGS)
	#$(CUDACC) $(CUDACFLAGS) -dc common_kernel.cu initialize.cu $(IFLAGS)
	#$(CUDACC) -arch=sm_30 -dlink common_kernel.o initialize.o -o $(OBJ_DIR)/initialize_linked.o
	#mv common_kernel.o $(OBJ_DIR)/common_kernel_inc.o
	#mv initialize.o $(OBJ_DIR)/initialize_inc.o

slab_config:
	$(CC) $(CFLAGS) -c -o obj/slab_config.o slab_config.cpp $(IFLAGS)

slab_cuda: slab_config initialize
	$(CC) $(CFLAGS) -c -o obj/slab_cuda.o obj/initialize.o slab_cuda.cpp $(IFLAGS)
#	$(CUDACC) $(CUDACFLAGS)  -o obj/slab_cuda.o slab_cuda.cpp $(IFLAGS)

tests: cuda_array2 slab_cuda
	$(MAKE) -C $(TEST_DIR)

clean:
	rm obj/*.o


