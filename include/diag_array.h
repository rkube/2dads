///
/// @detailed Array used in diagnostic functions
/// 
/// @detailed From array_base, but includes function to compute mean, fluctuations and stuff
/// @detailed diag_array is derived from a templated class. 
/// @detailed Thus all members of the base class are unknown to the compiler and we have to resort to an
/// @detailed ugly using hack
/// @detailed Paragraph 14.6/3 of the C++11 Standard:

#ifndef DIAG_ARRAY_H
#define DIAG_ARRAY_H

// Define periodic so array_base uses index wrapping
#define PERIODIC

#include <cstring>
#include <iostream>
#include <cassert>
#include <vector>
#include <thread>
#include <functional>
#include "error.h"
#include "bounds.h"
#include "2dads_types.h"
#include "array_base.h"
#include "cuda_array4.h"
// #include "cuda_operators.h"

using namespace std;


// Each thread computes total mean over [n_start : n_end-1] x [0:My]
template <class R, class T>
void thr_add_to_mean(R* array, T& result, int m_start, int m_end)
{
    const int Nx{int(array -> get_nx())};
    int n{0}, m{0};
    for(m = m_start; m < m_end; m++)
        for(n = 0; n < Nx; n++)
            result += (*array)(m, n);
}


template <class R, class T>
void thr_get_max(R* array, T& result, int m_start, int m_end)
{
    const int Nx{int(array -> get_nx())};
    int n{0}, m{0};
    T f_max{-1000.0};
    for(m = m_start; m < m_end; m++)
        for(n = 0; n < Nx; n++)
            f_max = ((*array)(m, n) > f_max ? (*array)(m, n) : f_max);
    result = f_max;
}


template <class R, class T>
void thr_get_min(R* array, T& result, int m_start, int m_end)
{
    const int Nx{int(array -> get_nx())};
    int n {0};
    int m {0};
    T f_min{1000.0};
    for(m = m_start; m < m_end; m++)
        for(n = 0; n < Nx; n++)
            f_min = ((*array)(m, n) < f_min ? (*array)(m, n) : f_min);
    result = f_min; 
}


// Thread kernel to compute the poloidal average (y-direction) 
// Each thread computes mean over [n_start : n_end-1] 
template <class R, class T>
void thr_pol_avg(R* array, T* profile, int n_start, int n_end)
{
    T pol_avg{0.0};
    int n{0}, m{0};
    const int My{int(array -> get_my())};
    for(n = n_start; n < n_end; n++)
    {
        pol_avg = 0.0;
        for(m = 0; m < My; m++)
        {
            pol_avg += (*array)(m, n);
        }
        profile[n] = pol_avg / T(My);
    }
}


// Set each y-value to profile[n]
template <class R, class T>
void thr_set_polavg(R* array, int n_start, int n_end)
{
    T pol_avg{0.0};
    int n{0}, m{0};
    const int My{int(array -> get_my())};

    for(n = n_start; n < n_end; n++)
    {
        pol_avg = 0.0;
        for(m = 0; m < My; m++)
            pol_avg += (*array)(m, n);
        pol_avg /= T(My);

        for(m = 0; m < My; m++)
            (*array)(m, n) = pol_avg;
    }
}


// Subtract y-average from each radial position
template <class R, class T>
void thr_set_tilde(R* array, int n_start, int n_end)
{
    T pol_avg{0.0};
    int n{0}, m{0};
    const int My{int(array -> get_my())};

    for(n = n_start; n < n_end; n++)
    {
        pol_avg = 0.0;
        for(m = 0; m < My; m++)
            pol_avg += (*array)(n, m);
        pol_avg /= T(My);

        for(m = 0; m < My; m++)
            (*array)(n, m) -= pol_avg;
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
        // Create dummy array Nx*My, nthreads, tlevs = 1
        diag_array(const uint, const uint);
        // Create array nthreads, tlevs, Nx, My
        diag_array(const uint, const uint, const uint, const uint);
        // Create array from cuda_array
        diag_array(cuda_array<T>&);
        // Create array from base class
        diag_array(const array_base<T, diag_array<T> >*);
        // Create from diag_Array
        diag_array(const diag_array<T>&);
        diag_array(diag_array<T>&&);

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
                for(m = 0; m < my; m++)                                                         
                {                                                                                            
                    for(n = 0; n < nx; n++)                                                     
                    {                                                                                        
                    os << src(t, m, n) << "\t";                                                            
                }                                                                                        
                os << "\n";                                                                              
            }                                                                                            
            }                                                                                                
            return (os);                                                                                     
        }                                          
        // Inline these definitions to avoid linker confusion when these guys
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

        inline void dump_profile() const;
        inline void dump_profile_t() const;

        inline void update(cuda_array<T>&);
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
diag_array<T> :: diag_array(cuda_array<T>& in) :
    array_base<T, diag_array<T>>(1, 1, in.get_my(), in.get_nx())
{
#ifdef DEBUG 
    cout << "diag_array::diag_array(cuda_array<T>& in)\n";
    cout << "\t\tarray: " << sizeof(T) * tlevs * Nx * My << " bytes at " << array << "\n"; 
    cout << "\tarray_t[0] at " << array_t[0] << "\n";
#endif
    size_t memsize = Nx * My;
    gpuErrchk(cudaMemcpy(array, in.get_array_d(), memsize * sizeof(T), cudaMemcpyDeviceToHost));
}

/// @brief Calls corresponding constructor from array_base
template <class T>
diag_array<T> :: diag_array(const array_base<T, diag_array<T>>* in) : array_base<T, diag_array<T>>(in) {}

template <class T>
diag_array<T> :: diag_array(const diag_array<T>& in) : array_base<T, diag_array<T>>(static_cast<const array_base<T, diag_array<T> >&> (in)) {}

template <class T>
diag_array<T> :: diag_array(diag_array<T>&& in) : array_base<T, diag_array<T>>(static_cast<array_base<T, diag_array<T> >&& >(in)) {}


/// @brief Calls corresponding constructor from array_base
template <class T>
diag_array<T> :: diag_array(uint My, uint Nx) :
    array_base<T, diag_array<T>>(1, 1, My, Nx)
{
}


/// @brief Calls corresponding constructor from array_base
/// @param nthreads Number of working threads for threaded member functions
/// @param tlevs: Number of time levels, use tlevs = 1
template <class T>
diag_array<T> :: diag_array(uint nthreads, uint tlevs, uint My, uint Nx) :
    array_base<T, diag_array<T>>(nthreads, tlevs, My, Nx)
{
}


/// @Copy data pointed to by in to memory localtion pointed to by array
/// @details assumes that nthreads=1, tlevs=1
template <class T>
void diag_array<T> :: update(cuda_array<T>& rhs)
{
    size_t line_size = My * Nx;
    check_bounds(rhs.get_tlevs(), rhs.get_my(), rhs.get_nx());
    gpuErrchk(cudaMemcpy(array, rhs.get_array_d(), line_size * sizeof(T), cudaMemcpyDeviceToHost));
//#ifdef DEBUG
//    cout << "diag_array::update(), host address: " << array << "\n";
//#endif
}

/*
 * ****************************************************************************
 * ****************************** Member functions ****************************
 * ****************************************************************************
 */


/// @brief Returns maximum of array
template<typename T>
inline T diag_array<T> :: get_max() const
{
    int n{0}, m{0};
    T f_max{-1.0};
    for(m = 0; m < int(My); m++)
        for(n = 0; n < int(Nx); n++)
            if(f_max < (*this)(m, n))
                f_max = (*this)(m, n);
    return(f_max);

}


template <typename T>
inline T diag_array<T> :: get_max_t() const
{
    unsigned int n{0};
    // Hold maxima found by each thread
    T result_nthreads[nthreads];
    for(n = 0; n < nthreads; n++)
        result_nthreads[n] = 0.0;

    // Spawn nthreads that compute the maximum over their domain
    std::vector<std::thread> thr;
    for(n = 0; n < nthreads; n++)
        thr.push_back(std::thread(thr_get_max<diag_array<T>, T>, const_cast<diag_array<T>*> (this), std::ref(result_nthreads[n]), n * nelem, (n + 1) * nelem));
    for(auto &t: thr)
        t.join();

    // Find maximum over maxima
    double f_max = -1000.0;
    for(n = 0; n < nthreads; n++)
        f_max = f_max > result_nthreads[n] ? f_max : result_nthreads[n];

    return f_max;
}



/// @brief return minimum value of array
template <typename T>
inline T diag_array<T> :: get_min() const
{
    int n{0}, m{0};
    T min{1e10};
    for(m = 0; m < int(My); m++)
        for(n = 0; n < int(Nx); n++)
            if(min > (*this)(m, n))
                min = (*this)(m, n);
    return(min);
}


template <typename T>
inline T diag_array<T> :: get_min_t() const
{
    unsigned int n{0};
    // Hold minima found by each threads
    T result_nthreads[nthreads];
    for(n = 0; n < nthreads; n++)
        result_nthreads[n] = 0.0;

    // Spawn nthreads threadsn that find the minimum over their domain
    std::vector<std::thread> thr;
    for(n = 0; n < nthreads; n++)
        thr.push_back(std::thread(thr_get_min<diag_array<T>, T>, const_cast<diag_array<T>*> (this), std::ref(result_nthreads[n]), n * nelem, (n + 1) * nelem));
    for(auto &t: thr)
        t.join();

    // Find maximum over maxima
    double f_min = 1000.0;
    for(n = 0; n < nthreads; n++)
        f_min = f_min < result_nthreads[n] ? f_min : result_nthreads[n];

    return f_min;
}
    

/// @brief Compute mean of array, normalized by Nx*My
template <class T> 
inline T diag_array<T> :: get_mean() const
{
    int n{0}, m{0};
    T mean{0.0};

    for(m = 0; m < int(My); m++)
        for(n = 0; n < int(Nx); n++)
            mean += (*this)(m, n);
    mean /= T(My * Nx);
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
            result += (*this)(m, n);
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


template <class T>
inline void diag_array<T> :: dump_profile() const
{
    T* profile = new T[Nx];
    get_profile(profile);
    for(int n = 0; n < int(Nx); n++)
        cout << profile[n] << "\t";
    delete[] profile;
}


template <class T>
inline void diag_array<T> :: dump_profile_t() const
{
    T* profile = new T[Nx];
    get_profile_t(profile);
    for(int n = 0; n < int(Nx); n++)
        cout << profile[n] << "\t";
    delete[] profile;
}


/// @brief Return an array with the radial profile, i.e. pol.avg at each radial position
template <class T>
inline diag_array<T> diag_array<T> :: bar() const
{
    diag_array<T> result(nthreads, 1, Nx, My);
    T temp {0.0};

    int n{0}, m{0};
    for(n = 0; n < int(Nx); n++)
    {
        temp = 0.0;
        for(m = 0; m < int(My); m++)
            temp += (*this)(n, m);
        temp = temp / T(My);
        for(m = 0; m < int(My); m++)
            result(m, n) = temp;
    }
    return(result);
}


template <class T>
inline diag_array<T> diag_array<T> :: bar_t() const
{
    // Result copies data from this, we will work on this array only
    diag_array<T> result(*this);
    int n{0};

    // Spawn threads that set result to poloidal average at each radial position
    std::vector<std::thread> thr;
    for(n = 0; n < nthreads; n++)
        thr.push_back(std::thread(thr_set_polavg<diag_array<T>, T>, &result, n * nelem, (n + 1) * nelem));
    for(auto &t: thr)
        t.join();
    
    return(result);
}


/// @brief Return copy of an array where the radial profile is subtracted at each n
template <class T>
inline diag_array<T> diag_array<T> :: tilde() const
{
    diag_array<T> result(*this);

    T* profile = new T[Nx];
    get_profile(profile);

    int n{0}, m{0};
    for(n = 0; n < int(Nx); n++)
    {
        //profile_n = this -> get_profile(n);
        for(m = 0; m < int(My); m++)
            result(m, n) -= profile[n];
    }
    delete[] profile;
    return(result);
}


/// @brief return fluctuations, threaded version
template <class T>
inline diag_array<T> diag_array<T> :: tilde_t() const
{
    // Result array we will operate on
    diag_array<T> result(*this);
    int n{0};

    // Spawn nthreads threads which subtract the poloidal average at each radial position
    std::vector<std::thread> thr;
    for(n = 0; n < nthreads; n++)
        thr.push_back(std::thread(thr_set_tilde<diag_array<T>, T>, &result, n * nelem, (n + 1) * nelem));
    for(auto &t: thr)
        t.join();

    return (result);
}


/// @brief Compute first x-derivative with a second order FD scheme
/// @detailed f'(x) = f(x_{i+1})  - f(x_{i-1}) / 2 \delta x
template <class T>
inline diag_array<T> diag_array<T> :: d1_dx1(const double Lx)
{
    diag_array<T> result(nthreads, 1, Nx, My);
    int n{0}, m{0};
    const double inv2dx{0.5 * double(Nx) / Lx};

    for(m = 0; m < My; m++)
        for(n = 0; n < Nx; n++)
            result(m, n) = ((*this)(m, n + 1) - (*this)(m, n - 1)) * inv2dx;

    return result;
}


/// @brief Compute first x-derivative with a second order FD scheme
/// @detailed f'(x) = f(x_{i+1})  - f(x_{i-1}) / 2 \delta x
template <class T>
inline diag_array<T> diag_array<T> :: d1_dy1(const double Ly)
{
    diag_array<T> result(nthreads, 1, Nx, My);
    int n{0}, m{0};
    const double inv2dy{0.5 * double(My) / Ly};

    for(m = 0; m < My; m++)
        for(n = 0; n < Nx; n++)
            result(n, m) = ((*this)(n, m + 1) - (*this)(n, m - 1)) * inv2dy;

    return result;
}


/// @brief Compute second x-derivative with a second order FD scheme
/// @brief f''(x) = f(x_{i-1}) - 2 f(x_{i}) + f(x_{i+1}) / delta_x^2
template <class T>
inline diag_array<T> diag_array<T> :: d2_dx2(const double Lx)
{
    diag_array<T> result(nthreads, 1, Nx, My);
    int n{0}, m{0};
    const double invdx2{double(Nx * Nx) / (Lx * Lx)};

    for(m = 0; m < (int)My; m++)
        for(n = 0; n < (int)Nx; n++)
            result(m, n) = ((*this)(m, n - 1) - 2.0 * (*this)(m, n) + (*this)(m, n + 1)) * invdx2;

    return result;
}


/// @brief Compute second y-derivative with a second order FD scheme
/// @brief f''(x) = f(x_{i-1}) - 2 f(x_{i}) + f(x_{i+1}) / delta_x^2
template <class T>
inline diag_array<T> diag_array<T> :: d2_dy2(const double Ly)
{
    diag_array<T> result(nthreads, 1, Nx, My);
    int n{0}, m{0};
    const double invdy2{double(My * My) / (Ly * Ly)};

    for(m = 0; m < int(My); m++)
        for(n = 0; n < int(Nx); n++)
            result(m, n) = ((*this)(m - 1, n) - 2.0 * (*this)(m, n) + (*this)(m + 1, n)) * invdy2;

    return result;
}


/// @brief Compute third x-derivative with a second order FD scheme
/// @detailed f'''(x) = -f(x_{i-2}) + 2 f(x_{i-1}) - 2 f(x_{i+1}) + f(x_{i+1}) / 2 delta_x^3 
template <class T>
inline diag_array<T> diag_array<T> :: d3_dx3(const double Lx)
{
    diag_array<T> result(nthreads, 1, Nx, My);
    int n{0}, m{0};
    const double inv2dx3{0.5 * double(Nx * Nx * Nx) / double(Lx * Lx * Lx)};

    for(m = 0; m < My; m++)
        for(n = 0; n < Nx; n++)
            result(m, n) = (-1.0 * (*this)(m, n - 2) + 2.0 * ((*this)(m, n - 1) - (*this)(m, n + 1)) + (*this)(m, n + 2)) * inv2dx3;
    return result;
}


/// @brief Compute third y-derivative with a second order FD scheme
/// @detailed f'''(x) = -f(x_{i-2}) + 2 f(x_{i-1}) - 2 f(x_{i+1}) + f(x_{i+1}) / 2 delta_x^3 
template <class T>
inline diag_array<T> diag_array<T> :: d3_dy3(const double Ly)
{
    diag_array<T> result(nthreads, 1, Nx, My);
    int n{0}, m{0};
    const double inv2dy3{0.5 * double(My * My * My) / double(Ly * Ly * Ly)};

    for(m = 0; m < My; m++)
        for(n = 0; n < Nx; n++)
            result(m, n) = (-1.0 * (*this)(m - 2, m) + 2.0 * ((*this)(m - 1, m) - (*this)(m + 1, n)) + (*this)(m + 2, n)) * inv2dy3;
    return result;
}


#endif //DIAG_ARRAY_H
