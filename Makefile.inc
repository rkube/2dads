CC	= /usr/bin/g++-4.8
#CFLAGS = -O0 -g -DDEBUG -std=c++11 -Wall -malign-double
CFLAGS = -O2 -std=c++11 -Wall -malign-double
IFLAGS = -I/home/rku000/cuda-workspace/cuda_array2/ -I/usr/local/cuda/include
LFLAGS = -L/usr/local/lib -L/usr/local/cuda/lib -lcudart -lcufft -lboost_program_options -lhdf5_hl_cpp -lhdf5_cpp -lhdf5  -lhdf5_hl

CUDACC	= /usr/local/cuda/bin/nvcc
#CUDACFLAGS	= -O0 -g -G -DDEBUG --gpu-architecture sm_30 -c --compiler-options -Wall
CUDACFLAGS	= -O3 --gpu-architecture sm_30 -c --compiler-options -Wall
CUDALFLAGS	= -lcufft

OBJ_DIR = /home/rku000/cuda-workspace/cuda_array2/obj/
