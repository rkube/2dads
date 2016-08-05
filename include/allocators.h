/*
 * Define allocator and deleter for device memory access
 */
#ifndef ALLOCATOR_DEVICE_H
#define ALLOCATOR_DEVICE_H

#include <memory>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "error.h"


template <typename T>
class my_deleter_host
{
    public:
        void operator()(T* p)
        {
            //std::cout << "deleter_host :: operator()" << std::endl;
            delete [] p;
            //std::cerr << "...memory deleted" << std::endl;
        }
};


template <typename T>
class my_deleter_device
{
    public:
        void operator() (T* p)
        {
            //std::cerr << "deleter_device :: operator()" << ", freeing at: " << p << std::endl;
            cudaError_t res;
            if ((res = cudaFree(static_cast<void*>(p))) != cudaSuccess)
            {
                throw gpu_error(cudaGetErrorString(res));
            }
            //std::cerr << "deleter_device :: freed memory" << std::endl;
        }
};


template <typename U>
struct my_allocator_device
{
    using ptr_type = std::unique_ptr<U, my_deleter_device<U> >;
    using value_type = U;

    public:

        // Delete the pointer using the deleter_device<U> class
        static void free(ptr_type ptr)
        {
            //std::cerr << "allocator_device :: free. Freeing at " << ptr.get() << std::endl;
            my_deleter_device<U> del;
            del(ptr.get());
            //std::cerr << "allocator_device :: free ... done" << std::endl;
        }

        // Allocate s * sizeof(U) bytes using cudaMalloc
        static ptr_type alloc(size_t s)
        {  
            void* ptr{nullptr};
            cudaError_t res;
            //std::cerr << "allocator_device :: alloc" << std::endl;
            if((res = cudaMalloc(&ptr, s * sizeof(U))) != cudaSuccess)
            {
                std::cerr << "Error allocating " << s * sizeof(U) << " bytes: " << cudaGetErrorString(res) << std::endl;
                throw;
            }
            //std::cerr << "allocated at " << ptr <<  std::endl;
            return ptr_type(static_cast<U*>(ptr));
        }

        // Copy (end-begin) elements from begin to dst
        static void copy(ptr_type begin, ptr_type end, ptr_type dst) {
            cudaMemcpy(dst.get(), begin.get(), (end.get() - begin.get()) * sizeof(U), cudaMemcpyDeviceToDevice);
        }

        // Make a copy of (end - begin) and return a pointer to it
        static ptr_type clone (ptr_type begin, ptr_type end)  { 
            ptr_type p = alloc(end - begin);
            copy(begin, end, p);
            return(p);
        }
};


template <typename U>
class my_allocator_host
{
    using ptr_type = std::unique_ptr<U, my_deleter_host<U> >;
    using value_type = U;

    public:
        // Delete the pointer using deleter_host<U>
        static void free(ptr_type ptr) 
        { 
            //std::cerr << "allocator_host :: free" << std::endl;
            my_deleter_host<U> del;
            del(ptr.get());
            //std::cerr << "allocator_host :: free ... done" << std::endl;
        }

        // Allocate s * sizeof(U) bytes
        static ptr_type alloc (size_t s) 
        { 
            //std::cerr << "allocator_host :: alloc" << std::endl;
            ptr_type ptr{new U[s]};
            return(ptr);
        } 

        static void copy (U* begin, U* end, U*dst)
        {
            std::copy ( begin, end, dst );
        }
        
        static U* clone (U* begin, U* end)
        {
            U* ptr = alloc(end - begin); 
            copy(begin, end, ptr);
            return(ptr);
        }    

};



template<class allocator>
struct my_allocator_traits
{
    //using allocator_type = allocator_device<U>;
    //using value_type = U;
    //using deleter_type = my_deleter_device<U>;
};


template <typename U>
struct my_allocator_traits<my_allocator_host<U> > 
{
    using allocator_type = my_allocator_host<U>;
    using value_type = U;
    using deleter_type = my_deleter_host<U>;
};



template <typename U>
struct my_allocator_traits<my_allocator_device<U> >
{
    using allocator_type = my_allocator_device<U>;
    using value_type = U;
    using deleter_type = my_deleter_device<U>;
};

#endif //ALLOCATOR_DEVICE_H
