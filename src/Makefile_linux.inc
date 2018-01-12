MKLROOT	= /home/rku000/local/intel/mkl
IOMPDIR = /home/rku000/local/intel/compilers_and_libraries/linux/lib/intel64

CC	= /home/rku000/local/bin/clang++

# Do not flag -pg when linking against gperftools as it messes stuff up
# https://github.com/gperftools/gperftools/issues/396
# Debug build
CFLAGS = -DDEBUG -DMKL_ILP64 -O0 -g -stdlib=libc++ -std=c++14 -Wall -fopenmp -fno-limit-debug-info
# Release build
#CFLAGS = -DMKL_ILP64 -O2  -stdlib=libc++ -std=c++14 -Wall -fopenmp 

# assume cuda is installed as a debian package in /usr/
INCLUDES = -I/home/rku000/source/2dads/src/include -I/home/rku000/local/include -I${MKLROOT}/include 

#LFLAGS = -L${MKLROOT}/lib -L/opt/local/lib -Wl,-rpath,${MKLROOT}/lib -lhdf5 -lfftw3  -lmkl_intel_ilp64 -lmkl_core -lmkl_intel_thread -liomp5 -lpthread -lm -ldl
LFLAGS = -L${MKLROOT}/lib/intel64 -L${IOMPDIR} -L/home/rku000/local/lib -Wl,-rpath,${MKLROOT}/lib -Wl,--no-as-needed -lhdf5 -lhdf5_cpp -lfftw3  -lmkl_intel_ilp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl

#NVCC	= /usr/local/cuda/bin/nvcc
CUDACC = /home/rku000/local/bin/clang++

#### Flags for NVCC
NVCCFLAGS	= -DDEBUG -DBOOST_NOINLINE='__attribute__ ((noinline))' -O0 -g -pg -G -lineinfo -std c++11 --gpu-architecture sm_30 -Xcompiler -Wall --expt-extended-lambda --compiler-bindir=/usr/bin  
#NVCCFLAGS	= -O3 -std c++11 --gpu-architecture sm_50 -Xcompiler -Wall --expt-extended-lambda --compiler-bindir=/usr/bin

### Flags when compiling device code with clang. Testing against clang 4.0.0
CUDACFLAGS	= -DDEBUG -O0 -g -std=c++14 -stdlib=libc++ --cuda-gpu-arch=sm_50 -Wall 
#CUDACFLAGS	= -O3 -stdlib=libc++ -std=c++14 --gpu-architecture sm_50 -Wall  

#CUDALFLAGS	= -L/usr/local/cuda/lib64 -lcufft -lcublas -lcusparse -lhdf5_serial -lhdf5_cpp -lpthread -lm -ldl
CUDALFLAGS	= -L/home/rku000/local/lib -lhdf5 -lhdf5_cpp -lcufft -lcublas -lcusparse -lcudart_static -ldl -lm -lpthread -lrt

LFLAGS_OGL = -lglfw -lGLEW -lGL

PROJ_INCDIR = /home/rku000/source/2dads/src/include
#OBJ_DIR = /home/rku000/source/2dads/src/obj/
OBJ_DIR = obj/

# MKL: file:///home/rku000/local/intel/documentation_2016/en/mkl/common/mkl_link_line_advisor.htm
# MKLROOT /home/rku000/local/intel
# linking:  -Wl,--no-as-needed -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl
# compiler options:  -DMKL_ILP64 -fopenmp -m64 -I${MKLROOT}/include
