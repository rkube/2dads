CC	= /usr/bin/g++-4.8
CFLAGS = -O0 -g -std=c++11 -Wall
IFLAGS = -I/home/rku000/cuda-workspace/cuda_array2/ -I/usr/local/cuda/include
LFLAGS = -L/usr/local/lib -L/usr/local/cuda/lib -lcudart -lcufft -lboost_program_options

CUDACC	= /usr/local/cuda/bin/nvcc
CUDACFLAGS	= -O0 -G --gpu-architecture sm_30 -c --compiler-options -Wall
CUDALFLAGS	= -lcufft

OBJ_DIR = /home/rku000/cuda-workspace/cuda_array2/obj/
