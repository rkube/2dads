#
include Makefile.inc
# Subdirectories
TEST_DIR = tests

.PHONY: slab_cuda clean tests

all: cuda_array2 output initialize diagnostics slab_config slab_cuda 
base: cuda_array2 initialize diagnostics slab_config output slab_cuda 


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
	$(CC) $(CFLAGS) -c -o obj/output.o output.cpp $(IFLAGS)

diagnostics:
	$(CC) $(CFLAGS) -c -o obj/diagnostics.o diagnostics.cpp $(IFLAGS)

#array_base:
#	$(CC) $(CFLAGS) -c -o obj/array_base.o array_base.cpp $(IFLAGS)
#diag_array: 
#	$(CC) $(CFLAGS) -c -o obj/diag_array.o diag_array.cpp $(IFLAGS)

tests: cuda_array2 slab_cuda initialize output
	$(MAKE) -C $(TEST_DIR)

clean:
	rm obj/*.o


