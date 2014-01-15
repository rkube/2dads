/*
 * array with additional member funcitons, used in diagnotics functions
 *
 */

#include "include/diag_array.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <thread>
#include <functional>

template <class T>
diag_array<T> :: diag_array(cuda_array<T>& in) :
    array_base<T>(1, 1, in.get_nx(), in.get_my())
{
#ifdef DEBUG 
    cout << "diag_array::diag_array(cuda_array<T>& in)\n";
    cout << "\t\tarray: " << sizeof(T) * array_base<T>::tlevs * array_base<T>::Nx * array_base<T>::My << " bytes at " << array_base<T>::array << "\n"; 
    cout << "\tarray_t[0] at " << array_base<T>::array_t[0] << "\n";
#endif
    size_t memsize = array_base<T>::Nx * array_base<T>::My;
    gpuErrchk(cudaMemcpy(array_base<T>::array, in.get_array_d(), memsize * sizeof(T), cudaMemcpyDeviceToHost));
}


template <class T>
void diag_array<T> :: update(cuda_array<T>& in)
{
    size_t memsize = array_base<T>::Nx * array_base<T>::My;
    gpuErrchk(cudaMemcpy(array_base<T>::array, in.get_array_d(), memsize * sizeof(T), cudaMemcpyDeviceToHost));
#ifdef DEBUG
    cout << "diag_array::update(), host address: " << array_base<T>::array << "\n";
#endif
}

/*
 * ****************************************************************************
 * ****************************** Operators ***********************************
 * ****************************************************************************
 */

template <class T>
diag_array<T>& diag_array<T> :: operator=(const T& rhs)
{
    uint n{0}, m{0};
    for(n = 0; n < array_base<T>::Nx; n++){
        for(m = 0; m < array_base<T>::My; m++){
            (*this)(0, n, m) = rhs;
        }
    }
    return *this;
}


template <class T>
diag_array<T>& diag_array<T> :: operator=(const diag_array<T>& rhs)
{
    // Sanity checks
    if(!array_base<T>::bounds(rhs.get_tlevs(), rhs.get_nx(), rhs.get_my()))
        throw out_of_bounds_err(string("diag_array<T>& diag_array<T>::operator=(const diag_array<T>& rhs): Index out of bounds: n\n"));
    if( (void*) this == (void*) &rhs)
        return *this;

    memcpy(array_base<T>::array, rhs.get_array(0), sizeof(T) * array_base<T>::Nx * array_base<T>::My);
    return *this;
}



// Compute the radial profile
//template <class T>
//void diag_array<T> :: update_profile()
//{
//    cout << "update profile\n";
//}


template class diag_array<twodads::real_t>;
