#
include Makefile_osx.inc
# Subdirectories
TEST_DIR = tests

.PHONY: clean dist

OBJECTS_HOST=$(OBJ_DIR)/slab_config.o $(OBJ_DIR)/output.o $(OBJ_DIR)/diagnostics_host.o $(OBJ_DIR)/slab_bc_host.o
OBJECTS_DEVICE=$(OBJ_DIR)/slab_config.o $(OBJ_DIR)/output.o $(OBJ_DIR)/diagnosics.o $(OBJ_DIR)/slab_bc_device.o

#shader.o: shader.cpp
#	$(CC) $(CFLAGS) $(DEFINES) -c -o $(OBJ_DIR)/shader.o shader.cpp $(INCLUDES)

slab_config_host.o: slab_config.cpp include/slab_config.h
	$(CC) $(CFLAGS) $(INCLUDES) -DHOST -c -o $(OBJ_DIR)/slab_config_host.o slab_config.cpp 

slab_config_device.o: slab_config.cpp include/slab_config.h
	$(CUDACC) $(CUDACFLAGS) $(INCLUDES) -DDEVICE -x cuda -c -o $(OBJ_DIR)/slab_config_device.o slab_config.cpp 


output.o: output.cpp include/output.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $(OBJ_DIR)/output.o output.cpp 

slab_bc_host.o: slab_bc.cpp include/slab_bc.h
	$(CC) $(CFLAGS) $(DEFINES) -DHOST $(INCLUDES) -c -o $(OBJ_DIR)/slab_bc_host.o slab_bc.cpp 

slab_bc_device.o: slab_bc.cu include/slab_bc.h
	$(CUDACC) $(NVCCFLAGS) $(DEFINES) -DDEVICE $(INCLUDES) -c -o $(OBJ_DIR)/slab_bc_device.o slab_bc.cu 

diagnostics_host.o: diagnostics.cpp include/diagnostics.h
	$(CC) $(CFLAGS) $(DEFINES) -DHOST $(INCLUDES) -c -o $(OBJ_DIR)/diagnostics_host.o diagnostics.cpp 

2dads_host: 
	$(CC) $(CFLAGS) -DHOST $(INCLUDES) -o run/2dads_host main_bc.cpp $(OBJECTS_HOST) $(LFLAGS)

# Profiling using gperftools
2dads_profile: 
	$(CC) $(CFLAGS) -DHOST $(INCLUDES) -o run/2dads_profile main_bc_profile.cpp $(OBJECTS_HOST) $(LFLAGS)  -ltcmalloc -lprofiler

2dads_device: 
#	$(CUDACC) $(CUDACFLAGS) -DDEVICE $(INCLUDES) -o run/2dads_device main_bc.cu $(OBJECTS_DEVICE) $(CUDALFLAGS)
	$(NVCC) $(NVCCFLAGS) -DDEVICE $(INCLUDES) -o run/2dads_device main_bc.cu $(OBJECTS_DEVICE) $(CUDALFLAGS)

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

