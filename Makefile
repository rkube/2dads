#
include Makefile.inc
# Subdirectories
TEST_DIR = tests

.PHONY: slab_config output diagnostics initialize slab_cuda tests clean dist 2dads

all: output initialize diagnostics slab_config slab_cuda shader
base: initialize diagnostics slab_config output slab_cuda shader

DEFINES	= -DPINNED_HOST_MEMORY -DBOOST_NOINLINE='__attribute__ ((noinline))' 


shader:
	$(CC) $(CFLAGS) $(DEFINES) -c -o $(OBJ_DIR)/shader.o shader.cpp $(INCLUDES)

slab_config:
	$(CC) $(CFLAGS) $(DEFINES) -c -o $(OBJ_DIR)/slab_config.o slab_config.cpp $(INCLUDES)

output: 
	$(CC) $(CFLAGS) $(DEFINES) -c -o $(OBJ_DIR)/output.o output.cpp $(INCLUDES)

diagnostics:
	$(CC) $(CFLAGS) $(DEFINES) -c -o $(OBJ_DIR)/diagnostics.o diagnostics.cpp $(INCLUDES)

initialize: 
	$(CUDACC) $(CUDACFLAGS) $(DEFINES) -c -o $(OBJ_DIR)/initialize.o initialize.cu $(INCLUDES)

slab_cuda:
	$(CUDACC) $(CUDACFLAGS) $(DEFINES) -c -o $(OBJ_DIR)/slab_cuda.o slab_cuda.cu $(INCLUDES)

2dads: slab_config output diagnostics initialize slab_cuda
	$(CC) $(CFLAGS) -o run/2dads $(OBJ_DIR)/slab_cuda.o $(OBJ_DIR)/slab_config.o $(OBJ_DIR)/initialize.o $(OBJ_DIR)/output.o $(OBJ_DIR)/diagnostics.o main.cpp $(INCLUDES) $(LFLAGS) 

tests: cuda_array2 slab_cuda initialize output diagnostics
	$(MAKE) -C $(TEST_DIR)

dist:
	rm -i dist/*.cpp
	rm -i dist/*.cu
	rm -i dist/include/*.h
	cp -R *.cpp *.cu dist
	cp Makefile dist/Makefile_nemesis
	cp Makefile.inc dist/Makefile_nemesis.inc
	cp include/*.h dist/include/
	cp tests/test_hw.cpp dist/main.cpp
	tar cvfj 2dads-cuda-`date +%Y-%m-%d`.tar.bz2 dist/*

clean:
	rm obj/*.o

