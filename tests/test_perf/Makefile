include ../../Makefile.inc
CUDACFLAGS	= -O0 -std c++11 --gpu-architecture sm_30 --compiler-options -Wall

SOURCES_CU = test_perf_derivs.cu

all: test_perf_derivs

test_perf_derivs_map: test_perf_derivs_map.cu
	$(CUDACC) $(CUDACFLAGS) -DDEBUG --generate-line-info -o test_perf_derivs_map test_perf_derivs_map.cu $(INCLUDES) $(CUDALFLAGS)

test_perf_derivs_comp: test_perf_derivs_comp.cu
	$(CUDACC) $(CUDACFLAGS) -DDEBUG --generate-line-info -o test_perf_derivs_comp test_perf_derivs_comp.cu $(INCLUDES) $(CUDALFLAGS)

test_derivs: test_perf_derivs_map2.cu
	$(CUDACC) $(CUDACFLAGS) -o test_perf_derivs_map2 test_perf_derivs_map2.cu $(INCLUDES) $(CUDALFLAGS)

