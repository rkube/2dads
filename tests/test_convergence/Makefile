include ../../Makefile.inc

SOURCES_CU = test_convergence.cu

test_convergence: test_convergence.cu $(PROJ_INCDIR)/cuda_array4.h
	$(CUDACC) $(CUDACFLAGS) -o test_convergence test_convergence.cu $(INCLUDES) $(LFLAGS)
