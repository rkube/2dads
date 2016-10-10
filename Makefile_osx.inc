CC	= /Users/ralph/local/bin/clang++
#CC	= /opt/local/bin/g++

CFLAGS = -DDEBUG -DMKL_ILP_64 -O0 -g -std=c++11 -Wall 
#CFLAGS = -O3 -march=native -std=c++11 -Wall

MKLROOT	= /opt/intel/mkl
IOMPDIR = /opt/intel/compilers_and_libraries_2016.3.170/mac/compiler/lib/

INCLUDES = -I/Users/ralph/source/2dads/include -I/usr/local/cuda/include -I${MKLROOT}/include -I/opt/local/include -I/Users/ralph/local/include

#LFLAGS = -L${MKLROOT}/lib -L/opt/local/lib -Wl,-rpath,${MKLROOT}/lib -lhdf5 -lfftw3  -lmkl_intel_ilp64 -lmkl_core -lmkl_intel_thread -liomp5 -lpthread -lm -ldl
#LFLAGS = -L${MKLROOT}/lib -L${IOMPDIR} -L/Users/ralph/local/lib -Wl,-rpath,${MKLROOT}/lib -lhdf5_cpp -lhdf5 -lhdf5_hl -lhdf5_hl_cpp -lfftw3  -lmkl_intel_ilp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -ldl
LFLAGS = -L/Users/ralph/local/lib -lhdf5_cpp -lhdf5 -lhdf5_hl -lhdf5_hl_cpp -lfftw3 -L${MKLROOT}/lib  -lmkl_intel_ilp64 -lmkl_core -lmkl_intel_thread -liomp5 -lpthread -lm -ldl

NVCC	= /Developer/NVIDIA/CUDA-8.0/bin/nvcc

NVCCFLAGS	= -DDEBUG -O0 -g -pg -G -lineinfo -std c++11 --gpu-architecture sm_30 -Xcompiler -Wall --expt-extended-lambda --compiler-bindir=/usr/bin  --ptxas-options="-v"
#NVCCFLAGS	= -O3 -std c++11 --gpu-architecture sm_30 -Xcompiler -Wall --expt-extended-lambda --compiler-bindir=/usr/bin

CUDALFLAGS	= -L/usr/local/cuda/lib -L/Users/ralph/local/lib -lcufft -lcublas -lcusparse -lhdf5_cpp -lhdf5 -lhdf5_hl -lhdf5_hl_cpp 

LFLAGS_OGL = -lglfw -lGLEW -lGL

PROJ_INCDIR = /Users/ralph/source/2dads/include
OBJ_DIR = /Users/ralph/source/2dads/obj

# MKL link line advisor
# https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor
# for osx with clang:
# link line:
#  -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_intel_ilp64 -lmkl_core -lmkl_intel_thread -liomp5 -lpthread -lm -ldl
# compiler options
#  -DMKL_ILP64 -m64 -I${MKLROOT}/include
