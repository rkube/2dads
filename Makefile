#
include Makefile.inc
# Subdirectories
TEST_DIR = tests

.PHONY: slab_cuda clean tests dist

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

tests: cuda_array2 slab_cuda initialize output diagnostics
	$(MAKE) -C $(TEST_DIR)

dist:
	cp -R *.cpp *.cu Makefile Makefile.inc include/ dist


clean:
	rm obj/*.o


