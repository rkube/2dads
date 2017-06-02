include ../../Makefile.inc

SOURCE_CPP = test_init_opengl.cpp test_diff_opengl1.cpp test_diff_opengl2.cpp

all: test_init_opengl test_diff_opengl1 test_diff_opengl2

DEP_OBJ = $(PROJ_OBJDIR)/initialize.o $(PROJ_OBJDIR)/slab_config.o $(PROJ_OBJDIR)/slab_cuda.o $(PROJ_OBJDIR)/shader.o $(PROJ_OBJDIR)/output.o

.PHONY: clean

test_init_opengl: test_init_opengl.cpp
	$(CC) $(CFLAGS) -o test_init_opengl $(DEP_OBJ) test_init_opengl.cpp $(INCLUDES) $(LFLAGS) $(LFLAGS_OGL)

test_diff_opengl1: test_diff_opengl1.cpp
	$(CC) $(CFLAGS) -o test_diff_opengl $(DEP_OBJ) test_diff_opengl1.cpp $(INCLUDES) $(LFLAGS) $(LFLAGS_OGL)

test_diff_opengl2: test_diff_opengl2.cpp
	$(CC) $(CFLAGS) -o test_diff_opengl2 $(DEP_OBJ) test_diff_opengl2.cpp $(INCLUDES) $(LFLAGS) $(LFLAGS_OGL)

clean:
	rm test_init_opengl test_diff_opengl1 test_diff_opengl2
