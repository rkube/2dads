include ../../Makefile.inc

SOURCE_CPP = output.cpp
all: test_output

DEP_OBJ = $(PROJ_OBJDIR)/output.o $(PROJ_OBJDIR)/slab_cuda.o $(PROJ_OBJDIR)/initialize.o $(PROJ_OBJDIR)/slab_config.o
.PHONY: clean

test_output: $(DEP_OBJ)
	$(CC) $(CFLAGS) -o test_output $(DEP_OBJ) test_output.cpp $(INCLUDES) $(LFLAGS)

clean:
	rm test_output


