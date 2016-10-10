#CC	= /usr/bin/g++
CC	= /Users/ralph/local/bin/clang++

## Do not use -pg when profiling with gperftools
#CFLAGS = -DDEBUG -O0 -pg -g -std=c++11 -Wall -malign-double -fopenmp
CFLAGS = -O2 -march=native -std=c++11 -Wall -malign-double 

INCLUDES = -I/Users/ralph/source/2dads/include -I/usr/local/cuda/include -I/opt/local/include/

#LFLAGS = -L/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu/hdf5/serial -L/usr/local/lib -L/usr/local/cuda/lib64 -lcudart -lcufft -lboost_program_options -lboost_filesystem -lboost_system -lhdf5_hl_cpp -lhdf5_cpp -lhdf5  -lhdf5_hl -lpthread -lfftw3 -lm
LFLAGS = -L/Users/ralph/local/intel/mkl/lib/intel64 -lboost_program_options -lboost_filesystem -lboost_system  -Wl,-rpath,/opt/local/lib/hdf5-18/lib  -lhdf5_hl_cpp -lhdf5_cpp -lhdf5 -lhdf5_hl -lfftw3 -lmkl_core -lmkl_intel_ilp64 -lmkl_gnu_thread -lpthread -lm 

CUDACC	= /usr/local/cuda/bin/nvcc
#CUDACFLAGS	= -O0 -g -pg -G -DDEBUG -lineinfo -std c++11 --gpu-architecture sm_50 -Xcompiler -Wall --expt-extended-lambda -lineinfo

CUDACFLAGS	= -O3 -std c++11 --gpu-architecture sm_30 --Xcompiler -Wall --expt-extended-lambda
CUDALFLAGS	= -L/usr/local/cuda/lib64 -lcufft -lcublas -lcusparse

LFLAGS_OGL = -lglfw -lGLEW -lGL

PROJ_INCDIR = /home/rku000/source/2dads/include
PROJ_OBJDIR = /home/rku000/source/2dads/obj

# MKL: file:///home/rku000/local/intel/documentation_2016/en/mkl/common/mkl_link_line_advisor.htm
# MKLROOT /home/rku000/local/intel
# linking:  -Wl,--no-as-needed -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl
# compiler options:  -DMKL_ILP64 -fopenmp -m64 -I${MKLROOT}/include
