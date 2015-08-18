/*
 *
 * Test inversion of Poisson Equation
 *
 */

#include <iostream>
#include <fstream>
#include <string>
#include <derivatives.h>
#include <cuda_array4.h>
#include <cuda_darray.h>

#ifdef __CUDA_ARCH__
#define CUDAMEMBER __device__ 
#endif

#ifndef __CUDA_ARCH__
#define CUDAMEMBER
#endif

using namespace std;


#define XNULL 1.1
#define YNULL 1.1

template <typename T>
class fun1 {
    // f(x, y) = exp(-(x - x_0)^2 - (y - y_0)^2)
    public:
        CUDAMEMBER T operator() (T x, T y) const { return ( exp(-(x - XNULL) * (x - XNULL) - (y - YNULL) * (y - YNULL)) );};
};





