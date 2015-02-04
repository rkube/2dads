include ../../Makefile.inc

SOURCES_CPP = test_blob.cpp test_hw.cpp test_zfky0.cpp test_zfky0_int.cpp

all: test_blob test_hw test_zfky0 
.PHONY: clean


test_blob: test_blob.cpp $(PROJ_OBJDIR)/slab_cuda.o  $(PROJ_OBJDIR)/slab_config.o  $(PROJ_OBJDIR)/output.o   $(PROJ_OBJDIR)/diagnostics.o   $(PROJ_OBJDIR)/initialize.o   
	$(CC) $(CFLAGS) -o test_blob $(PROJ_OBJDIR)/slab_cuda.o $(PROJ_OBJDIR)/slab_config.o $(PROJ_OBJDIR)/initialize.o $(PROJ_OBJDIR)/output.o $(PROJ_OBJDIR)/diagnostics.o test_blob.cpp $(INCLUDES) $(LFLAGS) -I/usr/local/cuda

test_hw: test_hw.cpp  $(PROJ_OBJDIR)/slab_cuda.o  $(PROJ_OBJDIR)/slab_config.o  $(PROJ_OBJDIR)/output.o   $(PROJ_OBJDIR)/diagnostics.o   $(PROJ_OBJDIR)/initialize.o   
	$(CC) $(CFLAGS) -o test_hw $(PROJ_OBJDIR)/slab_cuda.o $(PROJ_OBJDIR)/slab_config.o $(PROJ_OBJDIR)/initialize.o $(PROJ_OBJDIR)/output.o $(PROJ_OBJDIR)/diagnostics.o  test_hw.cpp $(INCLUDES) $(LFLAGS) -I/usr/local/cuda

test_zfky0: test_zfky0.cpp test_zfky0_int.cpp $(PROJ_OCJDIR)/slab_cuda.o  $(PROJ_OCJDIR)/slab_config.o  $(PROJ_OCJDIR)/output.o   $(PROJ_OCJDIR)/diagnostics.o   $(PROJ_OCJDIR)/initialize.o     
	$(CC) $(CFLAGS) -o test_zf/test_zfky0 ../obj/slab_cuda.o ../obj/slab_config.o ../obj/initialize.o ../obj/output.o ../obj/diagnostics.o test_zfky0.cpp $(INCLUDES) $(LFLAGS) -I/usr/local/cuda
	$(CC) $(CFLAGS) -o test_zf/test_zfky0_int ../obj/slab_cuda.o ../obj/slab_config.o ../obj/initialize.o ../obj/output.o ../obj/diagnostics.o test_zfky0_int.cpp $(INCLUDES) $(LFLAGS) -I/usr/local/cuda

clean:
	rm test_blob test_hw test_zfky0 test_zfky0_int

