#
include Makefile.inc
# Subdirectories
TEST_DIR = tests

.PHONY: slab_cuda clean tests

all: cuda_array2 output initialize slab_config slab_cuda
base: cuda_array2 initialize slab_config output slab_cuda

cuda_array2: 
	$(CUDACC) $(CUDACFLAGS) -o obj/cuda_array2.o cuda_array2.cu $(IFLAGS)

initialize: 
	$(CUDACC) $(CUDACFLAGS) -o obj/initialize.o initialize.cu $(IFLAGS)

slab_config:
	$(CC) $(CFLAGS) -c -o obj/slab_config.o slab_config.cpp $(IFLAGS)

slab_cuda:
	$(CC) $(CFLAGS) -c -o obj/slab_cuda.o slab_cuda.cpp $(IFLAGS)
	$(CUDACC) $(CUDACFLAGS) -o obj/slab_cuda2.o slab_cuda2.cu $(IFLAGS)

output: 
	$(CC) $(CFLAGS) -DDEBUG -c -o obj/output.o output.cpp $(IFLAGS)

tests: cuda_array2 slab_cuda initialize output
	$(MAKE) -C $(TEST_DIR)

clean:
	rm obj/*.o


