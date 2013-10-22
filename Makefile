#CC	= /opt/local/bin/g++-mp-4.8
CC	= /usr/bin/g++
CFLAGS = -O2 
#IFLAGS = -I/Users/ralph/source/cuda_test/include -I/Developer/NVIDIA/CUDA-5.5/include
#LFLAGS = -L/opt/local/lib -L/Developer/NVIDIA/CUDA-5.5/lib -lcudart -lcufft
IFLAGS = -I/home/rku000/cuda-workspace/cuda_array2/ -I/usr/local/cuda/include
LFLAGS = -L/usr/local/lib -L/usr/local/cuda/lib -lcudart -lcufft


CUDACC	= /usr/local/cuda/bin/nvcc
CUDACFLAGS	= -O2 --ptxas-options=-v --gpu-architecture sm_35
CUDALFLAGS	= -lcufft

.PHONY: cuda_array2 clean test_array


#test_array: cuda_array
#	$(CC) $(CFLAGS) -o test_array test_array.cpp obj/cuda_array.o $(IFLAGS) $(LFLAGS)
#
#test_dfts: cuda_array
#	$(CC) $(CFLAGS) -o test_dfts test_dfts.cpp obj/cuda_array.o $(IFLAGS) $(LFLAGS)
#
cuda_array2:
#	$(CC) $(CFLAGS) -c -o obj/cuda_array.o cuda_array.cpp $(IFLAGS) $(LFLAGS)
	$(CUDACC) $(CUDACFLAGS) -c -o obj/cuda_array2.o cuda_array2.cu 

test_array2: cuda_array2
	$(CC) $(CFLAGS) -o test_array2 obj/cuda_array2.o test_array2.cpp $(IFLAGS) $(LFLAGS) 

clean:
	rm obj/*.o


