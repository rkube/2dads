/*
 * Array used in diagnostic functions
 *
 * From array_base, but includes function to compute mean, fluctuatioons and stuff
 * diag_array is derived from a templated class. Thus all members of the base class
 * are unknown to the compiler.
 * Paragraph 14.6/3 of the C++11 Standard:
 */

#ifndef DIAG_ARRAY_H
#define DIAG_ARRAY_H

// Use periodic boundary conditions for array access
#define PERIODIC


#include <cstring>
#include <iostream>
#include <cassert>
#include <vector>
#include <thread>
#include <functional>
#include "error.h"
#include "check_bounds.h"
#include "2dads_types.h"
#include "array_base.h"
#include "cuda_array2.h"


using namespace std;


template <class T>
class diag_array : public array_base<T, diag_array<T> >
{
    using array_base<T, diag_array<T> >::My;
    using array_base<T, diag_array<T> >::Nx;
    using array_base<T, diag_array<T> >::tlevs;
    using array_base<T, diag_array<T> >:: array;
    using array_base<T, diag_array<T> >:: array_t;
    public:
        // Create dummy array Nx*My
        diag_array(uint, uint);
        // Create array from cuda_array
        diag_array(cuda_array<T>&);
        // Create array from base class
        diag_array(array_base<T, diag_array<T> >*);

        diag_array<T>& operator=(const T&);
        diag_array<T>& operator=(const diag_array<T>&);
        // Copy functions

        // Inlining operator<<, see http://stackoverflow.com/questions/4660123/overloading-friend-operator-for-template-class/4661372#4661372
        friend std::ostream& operator<<(std::ostream& os, const diag_array<T>& src)                             
        {                                                                                                    
            const int tl = int(src.get_tlevs());
            const int nx = int(src.get_nx());
            const int my = int(src.get_my());
            int t = 0, n = 0, m = 0;

            for(t = 0; t < tl; t++)                                                             
            {                                                                                                
                os << "t: " << t << "\n";                                                                    
                for(n = 0; n < nx; n++)                                                         
                {                                                                                            
                    for(m = 0; m < my; m++)                                                     
                    {                                                                                        
                    os << src(t,n,m) << "\t";                                                            
                }                                                                                        
                os << "\n";                                                                              
            }                                                                                            
            os << "\n\n";                                                                                
            }                                                                                                
            return (os);                                                                                     
        }                                          
        T get_mean() const;
        T get_max() const;
        T get_min() const;
        T get_profile(int) const;
        diag_array<T> bar() const;
        diag_array<T> tilde() const;


        void update(cuda_array<T>&);
        diag_array<T> d1_dx1(const diag_array<T>&, const double);
        diag_array<T> d2_dx2(const diag_array<T>&, const double);
        diag_array<T> d3_dx3(const diag_array<T>&, const double);
        diag_array<T> d1_dy1(const diag_array<T>&, const double);
        diag_array<T> d2_dy2(const diag_array<T>&, const double);
        diag_array<T> d3_dy3(const diag_array<T>&, const double);
};


template <class T>
diag_array<T> :: diag_array(cuda_array<T>& in) :
    array_base<T, diag_array<T>>(1, 1, in.get_nx(), in.get_my())
{
#ifdef DEBUG 
    cout << "diag_array::diag_array(cuda_array<T>& in)\n";
    cout << "\t\tarray: " << sizeof(T) * tlevs * Nx * My << " bytes at " << array << "\n"; 
    cout << "\tarray_t[0] at " << array_t[0] << "\n";
#endif
    size_t memsize = Nx * My;
    gpuErrchk(cudaMemcpy(array, in.get_array_d(), memsize * sizeof(T), cudaMemcpyDeviceToHost));
}


template <class T>
diag_array<T> :: diag_array(array_base<T, diag_array<T>>* in) : array_base<T, diag_array<T>>(in) {}

template <class T>
diag_array<T> :: diag_array(uint Nx, uint My) :
    array_base<T, diag_array<T>>(1, 1, Nx, My)
{
}


template <class T>
void diag_array<T> :: update(cuda_array<T>& in)
{
    size_t memsize = Nx * My;
    if(!(*this).bounds(in.get_tlevs(), in.get_nx(), in.get_my()))
        throw out_of_bounds_err(string("diag_array<T> :: update(cuda_array<T>& in): dimensions do not match!\n"));
    gpuErrchk(cudaMemcpy(array, in.get_array_d(), memsize * sizeof(T), cudaMemcpyDeviceToHost));
#ifdef DEBUG
    cout << "diag_array::update(), host address: " << array << "\n";
#endif
}

/*
 * ****************************************************************************
 * ****************************** Operators ***********************************
 * ****************************************************************************
 */

template<>
twodads::real_t diag_array<twodads::real_t> :: get_max() const
{
    int n{0}, m{0};
    twodads::real_t f_max{-1.0};
    for(n = 0; n < int(Nx); n++)
        for(m = 0; m < int(My); m++)
            if(f_max < (*this)(n, m))
                f_max = (*this)(n,m);
    return(f_max);
}

template <>
twodads::real_t diag_array<twodads::real_t> :: get_min() const
{
    int n{0}, m{0};
    twodads::real_t min{1e10};
    for(n = 0; n < int(Nx); n++)
        for(m = 0; m < int(My); m++)
            if(min > (*this)(n, m))
                min = (*this)(n,m);
    return(min);
}


template <class T> 
T diag_array<T> :: get_mean() const
{
    int n{0}, m{0};
    T mean{0.0};

    for(n = 0; n < int(Nx); n++)
        for(m = 0; m < int(My); m++)
            mean += (*this)(n,m);
    mean /= T(Nx * My);
    return(mean);
}

template <class T>
T diag_array<T> :: get_profile(int n) const
{
    T result{0.0};

    int m{0};
    for(m = 0; m < int(My); m++)
        result += (*this)(n, m);
    result /= T(My);
    return(result);
}


// Return an array with the radial profile, i.e. pol.avg at each radial position
template <class T>
diag_array<T> diag_array<T> :: bar() const
{
    diag_array<T> result(*this);
    T temp {0.0};

    int n{0}, m{0};
    for(n = 0; n < int(Nx); n++)
    {
        temp = 0.0;
        for(m = 0; m < int(My); m++)
            temp += (*this)(n, m);
        temp = temp / T(My);
        for(m = 0; m < My; m++)
            result(n, m) = temp;
    }
    return(result);
}



template <class T>
diag_array<T> diag_array<T> :: tilde() const
{
    diag_array<T> result(*this);

    T profile_n;
    int n{0}, m{0};
    for(n = 0; n < int(Nx); n++)
    {
        profile_n = this -> get_profile(n);
        for(m = 0; m < int(My); m++)
            result(n, m) -= profile_n;
    }
    return(result);
}



// d/dx derivative, second order scheme
template <class T>
diag_array<T> diag_array<T> :: d1_dx1(const diag_array<T>& rhs, const double Lx)
{
    //diag_array<T> result(array_base<T>::Nx, array_base<T>::My);
    diag_array<T> result(Nx, My);
    int n{0}, m{0};
    const double invdx2{Lx / (2.0 * double(Nx))};

    for(n = 0; n < Nx - 1; n++)
        for(m = 0; m < My; m++)
            result(n, m) = (rhs(n + 1, m) - rhs(n - 1, m)) * invdx2;

    return result;
}


template <class T>
diag_array<T> diag_array<T> :: d2_dx2(const diag_array<T>& rhs, const double Lx)
{
    diag_array<T> result(Nx, My);
    int n{0}, m{0};
    const double invdx{Lx / double(Nx)};


    for(n = 0; n < Nx; n++)
        for(m = 0; m < My; m++)
            result(n, m) = (rhs(n - 1, m) - 2.0 * rhs(n, m) + rhs(n + 1, m)) * invdx;

    return result;
}


template <class T>
diag_array<T> diag_array<T> :: d3_dx3(const diag_array<T>& rhs, const double Lx)
{
    diag_array<T> result(Nx, My);
    int n{0}, m{0};
    const double invdx2{0.5 * Lx / double(Nx)};

    for(n = 0; n < Nx; n++)
        for(m = 0; m < My; m++)
            result(n, m) = (-rhs(n - 2, m) + 2.0 * rhs(n - 1, m) - 2.0 * rhs(n + 1, m) + rhs(n + 2, m)) * invdx2;
    return result;
}


#endif //DIAG_ARRAY_H
