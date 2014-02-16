#
include Makefile.inc
# Subdirectories
TEST_DIR = tests

.PHONY: slab_config output diagnostics initialize slab_cuda tests clean dist

all: output initialize diagnostics slab_config slab_cuda 
base: initialize diagnostics slab_config output slab_cuda 


slab_config:
	$(CC) $(CFLAGS) -c -o obj/slab_config.o slab_config.cpp $(IFLAGS)

output: 
	$(CC) $(CFLAGS) -c -o obj/output.o output.cpp $(IFLAGS)

diagnostics:
	$(CC) $(CFLAGS) -c -o obj/diagnostics.o diagnostics.cpp $(IFLAGS)

initialize: 
	$(CUDACC) $(CUDACFLAGS) -c -o obj/initialize.o initialize.cu $(IFLAGS)

slab_cuda:
	$(CUDACC) $(CUDACFLAGS) -c -o obj/slab_cuda.o slab_cuda.cu $(IFLAGS)

tests: cuda_array2 slab_cuda initialize output diagnostics
	$(MAKE) -C $(TEST_DIR)

dist:
	cp -R *.cpp *.cu Makefile Makefile.inc include/ dist
	cp tests/test_hw.cpp dist/main.cpp
	mv dist/Makefile dist/Makefile_nemesis
	mv dist/Makefile.inc dist/Makefile_nemesis.inc

clean:
	rm obj/*.o

