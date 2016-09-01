MKLROOT	= /home/rku000/local/intel/mkl
#IOMPDIR = /home/rku000/local/intel/compilers_and_libraries_2016.3.170/mac/compiler/lib/

#CC	= /home/rku000/local/bin/clang++
CC = /usr/bin/g++

CFLAGS = -DDEBUG -O0 -pg -g -std=c++11 -Wall
#CFLAGS = -O3 -march=native -std=c++11 -Wall

INCLUDES = -I/home/rku000/source/cuda-workspace/2dads_remote/include -I/usr/local/cuda/include  -I${MKLROOT}/include -I/usr/include/hdf5/serial/

#LFLAGS = -L${MKLROOT}/lib -L/opt/local/lib -Wl,-rpath,${MKLROOT}/lib -lhdf5 -lfftw3  -lmkl_intel_ilp64 -lmkl_core -lmkl_intel_thread -liomp5 -lpthread -lm -ldl
LFLAGS = -L${MKLROOT}/lib/intel64 -L${IOMPDIR} -L/Users/ralph/local/lib -Wl,-rpath,${MKLROOT}/lib -lhdf5_cpp -lfftw3  -lmkl_intel_ilp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -ldl

CUDACC	= /usr/local/cuda/bin/nvcc

CUDACFLAGS	= -DDEBUG -DBOOST_NOINLINE='__attribute__ ((noinline))' -O0 -g -pg -G -lineinfo -std c++11 --gpu-architecture sm_30 -Xcompiler -Wall --expt-extended-lambda --compiler-bindir=/usr/bin  
#CUDACFLAGS	= -O3 -std c++11 --gpu-architecture sm_30 -Xcompiler -Wall --expt-extended-lambda --compiler-bindir=/usr/bin

#CUDALFLAGS	= -L/usr/local/cuda/lib64 -lcufft -lcublas -lcusparse -lhdf5_cpp -lhdf5 -lhdf5_hl -lhdf5_hl_cpp -lpthread -lm -ldl
CUDALFLAGS	= -L/usr/local/cuda/lib64 -lhdf5_serial -lhdf5_cpp -lcufft -lcublas -lcusparse -lpthread -lm -ldl

LFLAGS_OGL = -lglfw -lGLEW -lGL

PROJ_INCDIR = /home/rku000/source/cuda-workspace/2dads_remote/include
OBJ_DIR = obj/

# MKL: file:///home/rku000/local/intel/documentation_2016/en/mkl/common/mkl_link_line_advisor.htm
# MKLROOT /home/rku000/local/intel
# linking:  -Wl,--no-as-needed -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl
# compiler options:  -DMKL_ILP64 -fopenmp -m64 -I${MKLROOT}/include
