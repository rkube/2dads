#CC	= /opt/local/bin/g++-mp-4.8
CC	= /usr/bin/g++
CFLAGS = -O0 -g 
#IFLAGS = -I/Users/ralph/source/cuda_test/include -I/Developer/NVIDIA/CUDA-5.5/include
#LFLAGS = -L/opt/local/lib -L/Developer/NVIDIA/CUDA-5.5/lib -lcudart -lcufft
IFLAGS = -I/home/rku000/cuda-workspace/cuda_array2/ -I/usr/local/cuda/include
LFLAGS = -L/usr/local/lib -L/usr/local/cuda/lib -lcudart -lcufft


CUDACC	= /usr/local/cuda/bin/nvcc
CUDACFLAGS	= -O0 -G --ptxas-options=-v --gpu-architecture sm_30
CUDALFLAGS	= -lcufft

.PHONY: cuda_array2 clean test_array

all: cuda_array2 test_array2



#test_array: cuda_array
#	$(CC) $(CFLAGS) -o test_array test_array.cpp obj/cuda_array.o $(IFLAGS) $(LFLAGS)
#
test_dft: cuda_array2
	$(CC) $(CFLAGS) -o test_dft test_dft.cpp obj/cuda_array2.o $(IFLAGS) $(LFLAGS)

cuda_array2:
#	$(CC) $(CFLAGS) -c -o obj/cuda_array.o cuda_array.cpp $(IFLAGS) $(LFLAGS)
	$(CUDACC) $(CUDACFLAGS) -c -o obj/cuda_array2.o cuda_array2.cu 

test_array2: cuda_array2
	$(CC) $(CFLAGS) -o test_array2 obj/cuda_array2.o test_array2.cpp $(IFLAGS) $(LFLAGS) 

test_arrayc: cuda_array2
	$(CC) $(CFLAGS) -o test_arrayc obj/cuda_array2.o test_arrayc.cpp $(IFLAGS) $(LFLAGS)

clean:
	rm obj/*.o


