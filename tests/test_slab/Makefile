include ../../Makefile.inc

SOURCES_CPP: test_derivs_py.cpp test_derivs.cpp test_dft.cpp test_enum.cpp test_init.cpp
all: test_derivs test_dft test_enum test_init test_derivs_so

DEP_OBJ = $(PROJ_OBJDIR)/initialize.o $(PROJ_OBJDIR)/slab_config.o $(PROJ_OBJDIR)/slab_cuda.o
DEP_OBJPIC = $(PROJ_OBJDIR)/initialize_pic.o $(PROJ_OBJDIR)/slab_config_pic.o $(PROJ_OBJDIR)/slab_cuda_pic.o

.PHONY: clean
# Test enumeration of derivation kernels d_dx, d_dy, inv_lapl and time integrator stiff_k
test_enum: $(DEP_OBJ) test_enum.cpp
	$(CC) $(CFLAGS) -o test_enum $(PROJ_OBJDIR)/initialize.o $(PROJ_OBJDIR)/slab_config.o $(PROJ_OBJDIR)/slab_cuda.o test_enum.cpp $(INCLUDES) $(LFLAGS)

# Initialization
test_init: $(DEP_OPJ) test_init.cpp
	$(CC) $(CFLAGS) -o test_init $(PROJ_OBJDIR)/initialize.o $(PROJ_OBJDIR)/slab_config.o $(PROJ_OBJDIR)/slab_cuda.o test_init.cpp $(INCLUDES) $(LFLAGS)

# Test Fourier transformation 
test_dft: $(DEP_OBJ) test_dft.cpp
	$(CC) $(CFLAGS) -o test_dft $(PROJ_OBJDIR)/initialize.o  $(PROJ_OBJDIR)/slab_config.o  $(PROJ_OBJDIR)/slab_cuda.o test_dft.cpp $(INCLUDES) $(LFLAGS)

# Test that derivation routines work
test_derivs: $(DEP_OBJ)
	$(CC) $(CFLAGS) -o test_derivs $(PROJ_OBJDIR)/initialize.o $(PROJ_OBJDIR)/slab_config.o $(PROJ_OBJDIR)/slab_cuda.o test_derivs.cpp $(INCLUDES) $(LFLAGS)

test_derivs_so: $(DEP_OBJ_PIC)
	$(CC) $(CFLAGS) -shared -fPIC -Wl,-soname=test_derivs_py -o test_derivs_py.so $(PROJ_OBJDIR)/initialize_pic.o $(PROJ_OBJDIR)/slab_config_pic.o $(PROJ_OBJDIR)/slab_cuda_pic.o test_derivs_py.cpp $(INCLUDES) $(LFLAGS)



clean: 
	rm test_derivs test_dft test_enum test_init test_derivs_py.so
