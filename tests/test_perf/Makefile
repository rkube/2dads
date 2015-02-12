include ../../Makefile.inc

SOURCES_CU = test_perf_derivs.cu

all: test_perf_derivs

test_perf_derivs_map: test_perf_derivs_map.cu
	$(CUDACC) $(CUDACFLAGS) -DDEBUG --generate-line-info -o test_perf_derivs_map test_perf_derivs_map.cu $(INCLUDES) $(CUDALFLAGS)

test_perf_derivs_comp: test_perf_derivs_comp.cu
	$(CUDACC) $(CUDACFLAGS) -DDEBUG --generate-line-info -o test_perf_derivs_comp test_perf_derivs_comp.cu $(INCLUDES) $(CUDALFLAGS)

