/*
 * Define allocator and deleter for device memory access
 *
 * https://rawgit.com/google/cxx-std-draft/allocator-paper/allocator_user_guide.html
 *
 */
#ifndef ALLOCATOR_DEVICE_H
#define ALLOCATOR_DEVICE_H

#include <memory>
#include <iostream>
#include "error.h"

//#ifdef __CUDACC__
#if defined(__clang__) && defined(__CUDA__) && defined(__CUDA_ARCH__)
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif //__CUDACC__


//#ifdef __CUDACC__
#if defined(__clang__) && defined(__CUDA__) && defined(__CUDA_ARCH__)
template <typename T>
struct deleter_device
{
        void operator() (T* p)
        {
            //std::cerr << "deleter_device: freeing memory at " << p << std::endl;
            cudaError_t res;
            if ((res = cudaFree(static_cast<void*>(p))) != cudaSuccess)
            {
                throw gpu_error(cudaGetErrorString(res));
            }
        }
};


template <typename T>
struct allocator_device
{
    using ptr_type = std::unique_ptr<T, deleter_device<T> >;
    using deleter_type = deleter_device<T>;
    using value_type = T;

    allocator_device() noexcept{}
    template <class U> allocator_device(const allocator_device<U>&) noexcept{}

    template <class Other> struct rebind{using other = allocator_device<Other>;};

    // Allocate s * sizeof(T) bytes using cudaMalloc
    ptr_type allocate(size_t s)
    {  
        void* ptr{nullptr};
        cudaError_t res;
        if((res = cudaMalloc(&ptr, s * sizeof(T))) != cudaSuccess)
        {
            throw;
        }
        return ptr_type(static_cast<T*>(ptr));
    }

    // Delete the pointer using the deleter_device<T> class
    void deallocate(ptr_type ptr)
    {
        //std::cerr << "freeing memory at " << ptr.get() << std::endl;
        deleter_device<T> del;
        del(ptr.get());
    }

    // Copy (end-begin) elements from begin to dst
    void copy(ptr_type begin, ptr_type end, ptr_type dst) 
    {
        cudaMemcpy(dst.get(), begin.get(), (end.get() - begin.get()) * sizeof(T), cudaMemcpyDeviceToDevice);
    }

    void copy(T* begin, T* end, T* dst) 
    {
        cudaMemcpy(dst, begin, (end - begin) * sizeof(T), cudaMemcpyDeviceToDevice);
    }

    // Make a copy of (end - begin) and return a pointer to it
    ptr_type clone (ptr_type begin, ptr_type end)  
    { 
        ptr_type p = alloc(end - begin);
        copy(begin, end, p);
        return(p);
    }
};
#endif //__CUDACC__


template <typename T>
struct deleter_host
{
    void operator()(T* p) 
    { 
        //std::cerr << "deleter_host: freeing memory at " << p << std::endl;
        delete [] p; 
    }
};



template <typename T>
struct allocator_host
{
    using ptr_type = std::unique_ptr<T, deleter_host<T> >;
    using value_type = T;

    allocator_host() noexcept {}
    template <class U> allocator_host(const allocator_host<U>&) noexcept {}
    template <class Other> struct rebind{using other = allocator_host<Other>;};

    // Delete the pointer using deleter_host<T>
    void deallocate(ptr_type ptr) 
    { 
        //std::cerr << "freeing memory at " << ptr.get() << std::endl;
        deleter_host<T> del;
        del(ptr.get());
        //std::cerr << "allocator_host :: free ... done" << std::endl;
    }

    // Allocate s * sizeof(T) bytes
    ptr_type allocate (size_t s) 
    { 
        //std::cerr << "allocator_host :: allocating: " << s  << " * " << sizeof(T);
        ptr_type ptr;
        try
        {
            //ptr_type ptr{new T[s]};
            ptr.reset(new T[s]);
        }
        catch(std::exception& ba)
        {
            std::cerr << "bad_alloc caught: " << ba.what() << '\n';
        }
        //std::cerr << "\t...done. Allocated at " << ptr.get() << std::endl;
        return(ptr);
    } 

    void copy (T* begin, T* end, T*dst) { std::copy(begin, end, dst); }

    T* clone (T* begin, T* end)
    {
        T* ptr = alloc(end - begin); 
        copy(begin, end, ptr);
        return(ptr);
    }    
};


template<typename T, template <typename> class allocator>
struct my_allocator_traits
{
    using value_type = T;
};


template <typename T>
struct my_allocator_traits<T, allocator_host> 
{
    using allocator_type = allocator_host<T>;
    using value_type = T;
    using deleter_type = deleter_host<T>;
};


template <typename T>
struct my_allocator_traits<T*, allocator_host>
{
    using allocator_type = allocator_host<T*>;
    using value_type = T*;
    using deleter_type = deleter_host<T*>;
};


//#ifdef __CUDACC__
#if defined(__clang__) && defined(__CUDA__) && defined(__CUDA_ARCH__)
template <typename T>
struct my_allocator_traits<T, allocator_device>
{
    using allocator_type = allocator_device<T>;
    using value_type = T;
    using deleter_type = deleter_device<T>;
};


template <typename T>
struct my_allocator_traits<T*, allocator_device>
{
    using allocator_type = allocator_device<T*>;
    using value_type = T*;
    using deleter_type = deleter_device<T*>;
};
#endif //__CUDACC__


#endif //ALLOCATOR_DEVICE_H