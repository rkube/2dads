///
/// @detailed Array used in diagnostic functions
/// 
/// @detailed From array_base, but includes function to compute mean, fluctuations and stuff
/// @detailed diag_array is derived from a templated class. 
/// @detailed Thus all members of the base class are unknown to the compiler and we have to resort to an
/// @detailed ugly using hack
/// @detailed Paragraph 14.6/3 of the C++11 Standard:

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
#include "cuda_array3.h"


using namespace std;



#ifndef DIAG_ARRAY_H
#define DIAG_ARRAY_H

#define PERIODIC

// Each thread computes total mean over [n_start : n_end-1] x [0:My]
template <class R, class T>
void thr_add_to_mean(R* array, T& result, unsigned int n_start, unsigned int n_end)
{
    const unsigned int My{array -> get_my()};
    unsigned int n, m;
    for(n = n_start; n < n_end; n++)
        for(m = 0; m < My; m++)
            result += (*array)(n, m);
}


template <class R, class T>
void thr_get_max(R* array, T& result, unsigned int n_start, unsigned int n_end)
{
    const unsigned int My{array -> get_my()};
    unsigned int n, m;
    T f_max{-1.0};
    for(n = n_start; n < n_end; n++)
        for(m = 0; m < My; m++)
            f_max = ((*array)(n, m) > f_max ? (*array)(n, m) : f_max);
    result = f_max;
}


template <class R, class T>
void thr_get_min(R* array, T& result, unsigned int n_start, unsigned int n_end)
{
    const unsigned int My{array -> get_my()};
    unsigned int n {0.0};
    unsigned int m {0.0};
    T f_min{1000.0};
    for(n = n_start; n < n_end; n++)
        for(m = 0; m < My; m++)
            f_min = ((*array)(n, m) < f_min ? (array)(n, m) : f_min);
    result = f_min; 
}


// Thread kernel to compute the poloidal average (y-direction) 
// Each thread computes mean over [n_start : n_end-1] 
template <class R, class T>
void thr_pol_avg(R* array, T* profile, unsigned int n_start, unsigned int n_end)
{
    T pol_avg{0.0};
    unsigned int n, m;
    const unsigned int My{array -> get_my()};
    for(n = n_start; n < n_end; n++)
    {
        pol_avg = 0.0;
        for(m = 0; m < My; m++)
        {
            pol_avg += (*array)(n, m);
        }
        profile[n] = pol_avg / T(My);
    }
}



template <class T>
class diag_array : public array_base<T, diag_array<T> >
{
using array_base<T, diag_array<T> >::My;
using array_base<T, diag_array<T> >::Nx;
using array_base<T, diag_array<T> >::tlevs;
using array_base<T, diag_array<T> >::nthreads;
using array_base<T, diag_array<T> >::nelem;
using array_base<T, diag_array<T> >:: array;
using array_base<T, diag_array<T> >:: array_t;
public:
        // Create dummy array Nx*My
        diag_array(uint, uint);
        // Create array from cuda_array
        diag_array(cuda_array<T, T>&);
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
        // Inine these definitions to avoid linker confusion when these guys
        // pop up in multiple object files
        inline T get_mean() const;
        inline T get_mean_t() const;
        inline T get_max() const;
        inline T get_max_t() const;
        inline T get_min() const;
        inline T get_min_t() const;
        inline void get_profile(T*) const;
        inline void get_profile_t(T*) const;
        inline diag_array<T> bar() const;
        inline diag_array<T> bar_t() const;
        inline diag_array<T> tilde() const;
        inline diag_array<T> tilde_t() const;

        inline void update(cuda_array<T, T>&);
        inline diag_array<T> d1_dx1(const double);
        inline diag_array<T> d2_dx2(const double);
        inline diag_array<T> d3_dx3(const double);
        inline diag_array<T> d1_dy1(const double);
        inline diag_array<T> d2_dy2(const double);
        inline diag_array<T> d3_dy3(const double);
};


/// @brief Create diag_array from cuda array
/// @detailed Create diag_array with same dimensions as cuda_array
/// @detailed nthreads = 1, tlevs = 1
template <class T>
diag_array<T> :: diag_array(cuda_array<T, T>& in) :
    array_base<T, diag_array<T>>(1, 1, in.get_nx(), in.get_my())
{
#ifdef DEBUG 
    cout << "diag_array::diag_array(cuda_array<T, T>& in)\n";
    cout << "\t\tarray: " << sizeof(T) * tlevs * Nx * My << " bytes at " << array << "\n"; 
    cout << "\tarray_t[0] at " << array_t[0] << "\n";
#endif
    size_t memsize = Nx * My;
    gpuErrchk(cudaMemcpy(array, in.get_array_d(), memsize * sizeof(T), cudaMemcpyDeviceToHost));
}

/// @brief Calls corresponding constructor from array_base
template <class T>
diag_array<T> :: diag_array(array_base<T, diag_array<T>>* in) : array_base<T, diag_array<T>>(in) {}


/// @brief Calls corresponding constructor from array_base
template <class T>
diag_array<T> :: diag_array(uint Nx, uint My) :
    array_base<T, diag_array<T>>(1, 1, Nx, My)
{
}


/// @Copy data pointed to by in to memory localtion pointed to by array
/// @details assumes that nthreads=1, tlevs=1
template <class T>
void diag_array<T> :: update(cuda_array<T, T>& in)
{
    size_t memsize = Nx * My;
    if(!(*this).bounds(in.get_tlevs(), in.get_nx(), in.get_my()))
        throw out_of_bounds_err(string("diag_array<T> :: update(cuda_array<T, T>& in): dimensions do not match!\n"));
    gpuErrchk(cudaMemcpy(array, in.get_array_d(), memsize * sizeof(T), cudaMemcpyDeviceToHost));
#ifdef DEBUG
    cout << "diag_array::update(), host address: " << array << "\n";
#endif
}

/*
 * ****************************************************************************
 * ****************************** Member functions ****************************
 * ****************************************************************************
 */


/// @brief Returns maximum of array
template<>
inline twodads::real_t diag_array<twodads::real_t> :: get_max() const
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
inline twodads::real_t diag_array<twodads::real_t> :: get_max_t() const
{
    unsigned int n{0};
    // Hold maxima found by each thread
    twodads::real_t result_nthreads[nthreads];
    for(n = 0; n < nthreads; n++)
        result_nthreads[n] = 0.0;

    // Spawn nthreads that compute the maximum over their domain
    std::vector<std::thread> thr;
    for(n = 0; n < nthreads; n++)
        thr.push_back(std::thread(thr_get_max<diag_array<double>, double>, const_cast<diag_array<double>*> (this), std::ref(result_nthreads[n]), n * nelem, (n + 1) * nelem));
    for(auto &t: thr)
        t.join();

    // Find maximum over maxima
    double f_max = -100.0;
    for(n = 0; n < nthreads; n++)
         f_max = f_max > result_nthreads[n] ? f_max : result_nthreads[n];

    return f_max;
}



/// @brief return minimum value of array
template <>
inline twodads::real_t diag_array<twodads::real_t> :: get_min() const
{
    int n{0}, m{0};
    twodads::real_t min{1e10};
    for(n = 0; n < int(Nx); n++)
        for(m = 0; m < int(My); m++)
            if(min > (*this)(n, m))
                min = (*this)(n,m);
    return(min);
}


/// @brief Compute mean of array, normalized by Nx*My
template <class T> 
inline T diag_array<T> :: get_mean() const
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
inline T diag_array<T> :: get_mean_t() const
{
    unsigned int n{0}; 
    T mean{0.0};
    // Parallel version
    T result_threads[nthreads];
    for(n = 0; n < nthreads; n++)
        result_threads[n] = 0.0;

    std::vector<std::thread> thr;
    for(n = 0; n < nthreads; n++)
        thr.push_back(std::thread(thr_add_to_mean<diag_array<T>, T>, const_cast<diag_array<T>*>(this), std::ref(result_threads[n]), n * nelem, (n + 1) * nelem));
    for(auto &t: thr)
        t.join();

    for(n = 0; n < nthreads; n++)
        mean += result_threads[n];
    mean = mean / double(Nx * My);
    return(mean);
}


/// @brief Return mean along y-direction at n
template <class T>
inline void diag_array<T> :: get_profile(T* profile) const
{
    T result{0.0};

    int n {0};
    int m {0};
    for(n = 0; n < int(Nx); n++)
    {
        result = 0.0;
        for(m = 0; m < int(My); m++)
            result += (*this)(n, m);
        profile[n] = result / T(My);
    }
}


template <class T>
inline void diag_array<T> :: get_profile_t(T* profile) const
{
    unsigned int n{0};

    // Create threads that compute the profile on [n * nelem  : (n + 1) * nelem - 1]
    std::vector<std::thread> thr;
    for(n = 0; n < nthreads; n++)
        thr.push_back(std::thread(thr_pol_avg<diag_array<T>, T>, const_cast<diag_array<T>*>(this), profile, n * nelem, (n + 1) * nelem));
    for(auto &t: thr)
        t.join();   
}



/// @brief Return an array with the radial profile, i.e. pol.avg at each radial position
template <class T>
inline diag_array<T> diag_array<T> :: bar() const
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


/// @brief Return copy of an array where the radial profile is subtracted at each n
template <class T>
inline diag_array<T> diag_array<T> :: tilde() const
{
    diag_array<T> result(*this);

    T* profile = (T*) malloc(Nx * sizeof(T));
    get_profile(profile);

    int n{0}, m{0};
    for(n = 0; n < int(Nx); n++)
    {
        //profile_n = this -> get_profile(n);
        for(m = 0; m < int(My); m++)
            result(n, m) -= profile[n];
    }
    return(result);
}


/// @brief return fluctuations, threaded version
template <class T>
inline diag_array<T> diag_array<T> :: tilde_t() const
{
    diag_array<T> result(Nx, My);
    T* profile_vec = (T*) malloc(Nx * sizeof(T));


}


/// @brief Compute first x-derivative with a second order FD scheme
/// @detailed f'(x) = f(x_{i+1})  - f(x_{i-1}) / 2 \delta x
template <class T>
inline diag_array<T> diag_array<T> :: d1_dx1(const double Lx)
{
    diag_array<T> result(Nx, My);
    int n{0}, m{0};
    const double inv2dx{0.5 * double(Nx) / Lx};

    for(n = 0; n < Nx; n++)
        for(m = 0; m < My; m++)
            result(n, m) = ((*this)(n + 1, m) - (*this)(n - 1, m)) * inv2dx;

    return result;
}


/// @brief Compute first x-derivative with a second order FD scheme
/// @detailed f'(x) = f(x_{i+1})  - f(x_{i-1}) / 2 \delta x
template <class T>
inline diag_array<T> diag_array<T> :: d1_dy1(const double Ly)
{
    diag_array<T> result(Nx, My);
    int n{0}, m{0};
    const double inv2dy{0.5 * double(My) / Ly};

    for(n = 0; n < Nx; n++)
        for(m = 0; m < My; m++)
            result(n, m) = ((*this)(n, m + 1) - (*this)(n, m - 1)) * inv2dy;

    return result;
}


/// @brief Compute second x-derivative with a second order FD scheme
/// @brief f''(x) = f(x_{i-1}) - 2 f(x_{i}) + f(x_{i+1}) / delta_x^2
template <class T>
inline diag_array<T> diag_array<T> :: d2_dx2(const double Lx)
{
    diag_array<T> result(Nx, My);
    int n{0}, m{0};
    const double invdx2{double(Nx * Nx) / (Lx * Lx)};

    for(n = 0; n < Nx; n++)
        for(m = 0; m < My; m++)
            result(n, m) = ((*this)(n - 1, m) - 2.0 * (*this)(n, m) + (*this)(n + 1, m)) * invdx2;

    return result;
}


/// @brief Compute second y-derivative with a second order FD scheme
/// @brief f''(x) = f(x_{i-1}) - 2 f(x_{i}) + f(x_{i+1}) / delta_x^2
template <class T>
inline diag_array<T> diag_array<T> :: d2_dy2(const double Ly)
{
    diag_array<T> result(Nx, My);
    int n{0}, m{0};
    const double invdy2{double(My * My) / (Ly * Ly)};

    for(n = 0; n < Nx; n++)
        for(m = 0; m < My; m++)
            result(n, m) = ((*this)(n, m - 1) - 2.0 * (*this)(n, m) + (*this)(n, m + 1)) * invdy2;

    return result;
}


/// @brief Compute third x-derivative with a second order FD scheme
/// @detailed f'''(x) = -f(x_{i-2}) + 2 f(x_{i-1}) - 2 f(x_{i+1}) + f(x_{i+1}) / 2 delta_x^3 
template <class T>
inline diag_array<T> diag_array<T> :: d3_dx3(const double Lx)
{
    diag_array<T> result(Nx, My);
    int n{0}, m{0};
    const double inv2dx3{0.5 * double(Nx * Nx * Nx) / double(Lx * Lx * Lx)};

    for(n = 0; n < Nx; n++)
        for(m = 0; m < My; m++)
            result(n, m) = (-1.0 * (*this)(n - 2, m) + 2.0 * ((*this)(n - 1, m) - (*this)(n + 1, m)) + (*this)(n + 2, m)) * inv2dx3;
    return result;
}


/// @brief Compute third y-derivative with a second order FD scheme
/// @detailed f'''(x) = -f(x_{i-2}) + 2 f(x_{i-1}) - 2 f(x_{i+1}) + f(x_{i+1}) / 2 delta_x^3 
template <class T>
inline diag_array<T> diag_array<T> :: d3_dy3(const double Ly)
{
    diag_array<T> result(Nx, My);
    int n{0}, m{0};
    const double inv2dy3{0.5 * double(My * My * My) / double(Ly * Ly * Ly)};

    for(n = 0; n < Nx; n++)
        for(m = 0; m < My; m++)
            result(n, m) = (-1.0 * (*this)(n, m - 2) + 2.0 * ((*this)(n, m - 1) - (*this)(n, m + 1)) + (*this)(n, m + 2)) * inv2dy3;
    return result;
}


#endif //DIAG_ARRAY_H
