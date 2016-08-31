#
include Makefile_osx.inc
# Subdirectories
TEST_DIR = tests

.PHONY: clean dist


DEFINES	= -DPINNED_HOST_MEMORY -DBOOST_NOINLINE='__attribute__ ((noinline))' 

OBJECTS_HOST=$(OBJ_DIR)/slab_config.o $(OBJ_DIR)/output.o $(OBJ_DIR)/slab_bc_host.o
OBJECTS_DEVICE=$(OBJ_DIR)/slab_config.o $(OBJ_DIR)/output.o $(OBJ_DIR)/slab_bc_device.o

#shader.o: shader.cpp
#	$(CC) $(CFLAGS) $(DEFINES) -c -o $(OBJ_DIR)/shader.o shader.cpp $(INCLUDES)

slab_config.o: slab_config.cpp include/slab_config.h
	$(CC) $(CFLAGS) $(DEFINES) -c -o $(OBJ_DIR)/slab_config.o slab_config.cpp $(INCLUDES)

output.o: output.cpp include/output.h
	$(CC) $(CFLAGS) $(DEFINES) -c -o $(OBJ_DIR)/output.o output.cpp $(INCLUDES)

slab_bc_host.o: slab_bc.cpp
	$(CC) $(CFLAGS) $(DEFINES) -DHOST -c -o $(OBJ_DIR)/slab_bc_host.o slab_bc.cpp $(INCLUDES)

slab_bc_device.o: slab_bc.cu
	$(CUDACC) $(CUDACFLAGS) $(DEFINES) -DDEVICE -c -o $(OBJ_DIR)/slab_bc_device.o slab_bc.cu $(INCLUDES)

#diagnostics.o: diagnostics.cpp include/diagnostics.h
#	$(CC) $(CFLAGS) $(DEFINES) -c -o $(OBJ_DIR)/diagnostics.o diagnostics.cpp $(INCLUDES)
#
#diagnostics_cu.o: diagnostics_cu.cu include/diagnostics_cu.h include/cuda_darray.h include/cuda_array4.h
#	$(CUDACC) $(CUDACFLAGS) $(DEFINES) -c -o $(OBJ_DIR)/diagnostics_cu.o diagnostics_cu.cu $(INCLUDES)
#
#2dads_profile: slab_config.o initialize.o slab_cuda.o
#	$(CC) $(CFLAGS) -o run/2dads_profile $(OBJECTS) main.cpp $(INCLUDES) $(LFLAGS) 
#	#$(CUDACC) $(CUDACFLAGS) -o run/2dads_profile $(OBJECTS) main.cpp $(INCLUDES) $(LFLAGS) 

2dads_host: 
	$(CC) $(CFLAGS) -DHOST -o run/2dads_host $(OBJECTS_HOST) main_bc.cpp $(INCLUDES) $(LFLAGS)

2dads_device: 
	$(CUDACC) $(CUDACFLAGS) -DDEVICE -o run/2dads_device $(OBJECTS_DEVICE) main_bc.cu $(INCLUDES) $(CUDALFLAGS)

dist:
	rm -i dist/*.cpp
	rm -i dist/*.cu
	rm -i dist/include/*.h
	cp -R *.cpp *.cu dist
	cp Makefile dist/Makefile_osx
	cp Makefile_osx.inc dist/Makefile_osx.inc
	cp include/*.h dist/include/
	tar cvfj 2dads-cuda-`date +%Y-%m-%d`.tar.bz2 dist/*

clean:
	rm obj/*.o

