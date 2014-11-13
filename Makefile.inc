CC	= /usr/bin/g++-4.8
#CFLAGS = -O0 -pg -g -DDEBUG -std=c++11 -Wall -malign-double
CFLAGS = -O2 -march=native -std=c++11 -Wall -malign-double
INCLUDES = -I/home/rku000/cuda-workspace/cuda_array2/include -I/usr/local/cuda/include
LFLAGS = -L/usr/local/lib -L/usr/local/cuda/lib64 -lcudart -lcufft -lboost_program_options -lhdf5_hl_cpp -lhdf5_cpp -lhdf5  -lhdf5_hl -lpthread -lm

CUDACC	= /usr/local/cuda/bin/nvcc
#CUDACFLAGS	= -O0 -g -pg -G -DDEBUG --gpu-architecture sm_30 --compiler-options -Wall
CUDACFLAGS	= -O2 --gpu-architecture sm_30 --compiler-options -Wall 
CUDALFLAGS	= -lcufft

OBJ_DIR = /home/rku000/cuda-workspace/cuda_array2/obj/
