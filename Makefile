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

initialize: 
	$(CUDACC) $(CUDACFLAGS) -o obj/initialize.o initialize.cu $(IFLAGS)

slab_config:
	$(CC) $(CFLAGS) -c -o obj/slab_config.o slab_config.cpp $(IFLAGS)

slab_cuda: slab_config initialize
	$(CC) $(CFLAGS) -c -o obj/slab_cuda.o slab_cuda.cpp $(IFLAGS)
	$(CUDACC) $(CUDACFLAGS) -o obj/slab_cuda2.o slab_cuda2.cu $(IFLAGS)

tests: cuda_array2 slab_cuda
	$(MAKE) -C $(TEST_DIR)

clean:
	rm obj/*.o


