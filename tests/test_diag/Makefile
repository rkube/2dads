include ../../Makefile.inc

OBJECTS=$(PROJ_OBJDIR)/slab_config.o $(PROJ_OBJDIR)/diagnostics.o $(PROJ_OBJDIR)/diagnostics_cu.o $(PROJ_OBJDIR)/initialize.o $(PROJ_OBJDIR)/slab_cuda.o


all: test_diag
test_diag: test_diag.cpp $(PROJ_INCDIR)/cuda_darray.h $(PROJ_INCDIR)/diagnostics_cu.h $(PROJ_OBJDIR)/diagnostics_cu.o
	$(CC) $(CFLAGS) -DDEBUG -o test_diag test_diag.cpp $(OBJECTS) $(INCLUDES) $(LFLAGS)


