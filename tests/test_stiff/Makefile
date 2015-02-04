include ../../Makefile.inc

SOURCE_CPP = test_diff.cpp test_stiff.cpp test_stiff_transient.cpp

all: test_diff test_stiff test_stiff_transient

DEP_OBJ = $(PROJ_OBJDIR)/initialize.o $(PROJ_OBJDIR)/slab_config.o $(PROJ_OBJDIR)/slab_cuda.o $(PROJ_OBJDIR)/output.o

.PHONY: clean

test_diff: test_diff.cpp $(DEP_OBJ)
	$(CC) $(CFLAGS) -o test_diff $(DEP_OBJ) test_diff.cpp $(INCLUDES) $(LFLAGS)

test_stiff: test_stiff.cpp $(DEP_OBJ)
	$(CC) $(CFLAGS) -o test_stiff $(DEP_OBJ) test_stiff.cpp $(INCLUDES) $(LFLAGS)

test_stiff_transient: test_stiff_transient.cpp $(DEP_OBJ)
	$(CC) $(CFLAGS) -o test_stiff_transient $(DEP_OBJ) test_stiff_transient.cpp $(INCLUDES) $(LFLAGS)

clean:
	rm test_diff test_stiff test_stiff_transient

