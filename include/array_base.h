/*
 * Header for array template class
 */
#ifndef ARRAY_BASE_H
#define ARRAY_BASE_H

#include <iostream>
#include <complex>
#include <cassert>
#include <vector>
#include <thread>
#include <cmath>
#include <cstring>
#include "error.h"
#include "check_bounds.h"
#include "2dads_types.h"

// Template for the loop operator used in operator[+-*]=
//
// Here R, i.e. array_base<double>, array_base<my_cmplex>,
// Functor is a function that takes two arguments of the base type of R,
// i.e. double, complex and returns a reference to the operators,
// see std::plus<T>, std::minus<T>, std::multiplies<T>, etc


template<class R, typename Functor>
#ifdef PERIODIC
inline void calc_loop(R* lhs, const R* rhs, int n_start, int n_end) 
#endif
#ifndef PERIODIC
inline void calc_loop(R* lhs, const R* rhs, unsigned int n_start, unsigned int n_end) 
#endif
{
    Functor op;
    //unsigned int My{rhs -> get_my()};
    
    //cout << "\tcalc_loop from n = " << n_start << "..." << n_end << "\n";
#ifdef PERIODIC
    int n = n_start;
    int m = 0;
    const int My = int(rhs -> get_my());
#endif
#ifndef PERIODIC
    unsigned int n = n_start;
    unsigned int m = 0;
    const uint My = rhs -> get_my();
#endif

    for(n = n_start; n < n_end; n++)
        for(m = 0; m < My; m++)
            (*lhs)(0, n, m) =  op( (*lhs)(0, n, m), (*rhs)(0, n, m));
}

// Same as above, with a scalar as the second argument
template<class R, class S, typename Functor>
#ifdef PERIODIC
inline void calc_loop_scalar(R* lhs, const S& rhs, int n_start, int n_end)
#endif
#ifndef PERIODIC
inline void calc_loop_scalar(R* lhs, const S& rhs, unsigned int n_start, unsigned int n_end)
#endif
{
    Functor op;

#ifdef PERIODIC
    int n = n_start;
    int m = 0;
    const int My = int(rhs -> get_my());
#endif
#ifndef PERIODIC
    unsigned int n = n_start;
    unsigned int m = 0;
    const uint My = lhs -> get_my();
#endif    
    for(n = n_start; n < n_end; n++)
        for(m = 0; m < My; m++)
            (*lhs)(0, n, m) = op( (*lhs)(0, n, m), rhs);
}


// Template parameters:
// T: template type of the derived class, i.e. double, complex, int, etc
// derived: type of the derived class. We use this, so that the return type of the 
// operators defined in this class return types of the derived class calling the methods,
// and no array_base.
//
// See https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern

template <class T, class Derived> 
class array_base{
    public:
        // C++11
        //using uint = unsigned int;
        typedef unsigned int uint;
        // C++11
        //typedef T* iterator;
        // C++11
        //using iterator = T*;
        // C++11
        //using const_iterator = const T*;
        // Define constructors 
        // No default constructor, dimensions need to be bassed on creation
        // C++11
        //array_base() = delete;
        // Copy constructor
        array_base(const array_base<T, Derived>&);
        array_base(array_base<T, Derived>*);
        // Move constructor
        //array_base(array_base<T>&&);
        // Create array with tlevs = 1
        array_base(const uint, const uint, const uint, const uint);
        array_base(const uint, const uint, const uint);
        array_base(const uint, const uint);
        // Destructor
        ~array_base();


        // Iterators
        //iterator begin() { return &array[0];}
        //iterator end() {return &array[Nx * My + 0];}

        // Operators
        // use -DPERIODIC, to wrap indices on array boundaries, i.e. n = -1 -> Nx - 1, etc...
#ifdef PERIODIC
        T& operator()(const uint, int, int);
        T operator() (const uint, int, int) const;

        T& operator()(int, int);
        T operator() (int, int) const;
#endif // PERIODIC
        // Use const uint as input since nx... is not modified in operator() implementation
#ifndef PERIODOC
        T& operator()(const uint, const uint, const uint);
        T operator() (const uint, const uint, const uint) const;

        T& operator()(const uint, const uint);
        T operator() (const uint, const uint) const;
#endif

        Derived& operator=(const T&);
        Derived& operator=(const array_base<T, Derived>&);
        // C++11
        //array_base<T>& operator=(array_base<T>&&);

        // These return the derived type but may take different derived types as input
        Derived& operator+=(const Derived&);
        Derived& operator+=(const T&);
        Derived operator+(const Derived&) const;
        Derived operator+(const T&) const;

        Derived& operator*=(const Derived&);
        Derived& operator*=(const T&);
        Derived operator*(const Derived&) const;
        Derived operator*(const T&) const;

        Derived& operator-=(const Derived&);
        Derived& operator-=(const T&);
        Derived operator-(const Derived&) const;
        Derived operator-(const T&) const;

        // Copy functions
        void advance();
        void copy(const uint, const uint);
        void copy(const uint, const Derived&, const uint);
        void move(const uint, const uint);

        // Inlining operator<<, see http://stackoverflow.com/questions/4660123/overloading-friend-operator-for-template-class/4661372#4661372
        friend std::ostream& operator<<(std::ostream& os, const array_base& src)
        {
            const uint tl = src.get_tlevs();
            const uint nx = src.get_nx();
            const uint my = src.get_my();

            for(unsigned int t = 0; t < tl; t++)
            {
                os << "t: " << t << "\n";
                for(unsigned int n = 0; n < nx; n++)
                {
                    for(unsigned int m = 0; m < my; m++)
                    {
                        os << src(t,n,m) << "\t";
                    }
                    os << "\n";
                }
                os << "\n\n";
            }
            return (os);
        }


        // Bounds function
        // Return the array element specified by (n,m)
        // time indexing is handled only via array_t
        inline int address(const uint n, const uint m) const {return(n * My + m);}

        // Access to dimensions
        inline  uint get_tlevs() const {return tlevs;}
        inline  uint get_nx() const {return Nx;}
        inline  uint get_my() const {return My;}
        inline T* get_array() const {return array;}
        inline T* get_array(uint t) const {return array_t[t];}
    protected:
        // number of threads to use
        uint nthreads;
        uint nelem;
        // Array bounds
        const uint tlevs;
        const uint Nx;
        const uint My;
        // Check for bounds
        check_bounds bounds;
        // Actual array data
        T* array;
        T** array_t;
};

/*
 * Template declarations of array_base
 *
 * This is the implementation of the template code. 
 * Template code is generated once the template is instantiated. 
 * Putting the complete template definition allows us to include only this header file
 * in the derived classes instead of using explicit template instantiation and linking.
 */

// Standard constructor for class array_base<T>
// Initialize array boundaries, allocate memory and initialize
// threading data
template<class T, class Derived>
array_base<T, Derived> :: array_base(uint nthr, uint t, uint n, uint m) :
    nthreads(nthr), 
    tlevs(t),
    Nx(n),
    My(m),
    bounds(tlevs, Nx, My),
    array(nullptr),
    array_t(nullptr)
{
    // Allocate memory
    array = (T*) malloc(sizeof(T) * tlevs * Nx * My);
    array_t = (T**) malloc(sizeof(T*) * tlevs);
    for(uint tl = 0; tl < tlevs; tl++)
        array_t[tl] = &array[tl * Nx * My];
    
    // Only allow a number of threads that divides 
    // Nx. If not, abort.
    assert(Nx % nthreads == 0);
    nelem = Nx / nthreads;

#ifdef DEBUG 
    cout << "array_base<T>::array_base(uint, uint, uint)\n";
    cout << "\t\ttlevs = " << tlevs << "\tNx = " << Nx << "\tMy = " << My << "\n";
    cout << "\t\tarray: " << sizeof(T) * tlevs * Nx * My << " bytes at " << array << "\n";
    cout << "\t\tarray_t: " << sizeof(T*) * tlevs << " bytes at " << array_t << "\n";
    for(uint tl = 0; tl < tlevs; tl++)
        cout << "\t\tarray_t[" << tl << "] at " << array_t[tl] << "\n";
    cout << "\t\tnthreads = " << nthreads << " threads, nelem = " << nelem << "\n";
#endif 
}


template<class T, class Derived>
array_base<T, Derived> :: array_base(uint n, uint m) :
    array_base(1, 1, n, m)
{
}


// Copy constructor calls the base constructor and copies 
// T* array from src to dst
template<class T, class Derived>
array_base<T, Derived> :: array_base(const array_base<T, Derived>& src) :
    array_base(src.nthreads, src.tlevs, src.Nx, src.My)
{
    memcpy(array, src.array, sizeof(T) * tlevs * Nx * My);
}


template<class T, class Derived>
array_base<T, Derived> :: array_base(array_base<T, Derived>* src) :
    array_base((*src).nthreads, (*src).tlevs, (*src).Nx, (*src).My)
{
    memcpy(array, (*src).array, sizeof(T) * tlevs * Nx * My);
}
// Move constructor
// Takes a rvalue as a reference and moves the data of this object into its
// own private variables.
// This is called in f.ex.:
// array_base<twodads::real_t> a1(t, N, M);
// array_base<twodads::real_t> a2(t, N, M);
// a1 = 1.0;
// a2 = 2.0;
// array_base<twodads::real_t> a3(a1 + a2);
// The constructor just copies the data of the temporary object "a1 + a2" 
//
// src is not const in this case
// Implement swap semantics for points to array and array_t between
// callar and RHS
//
// C++ 11 feature, inactive but tested
//template<class T>
//array_base<T> :: array_base(array_base<T>&& rhs) :
//    nthreads(rhs.nthreads),
//    nelem(rhs.nelem),
//    tlevs(rhs.tlevs),
//    Nx(rhs.Nx), 
//    My(rhs.My),
//    bounds(tlevs, Nx, My)
//{
//    //cout << "array_base<T>::array_base(const array_base<T>&& src)\tMove constructor\n";
//   
//    T* tmp_array = array;
//    T** tmp_array_t = array_t;
//    array = rhs.array;
//    array_t = rhs.array_t;
//    rhs.array = tmp_array;
//    rhs.array_t = tmp_array_t;
//    //cout << "done swapping\n";
//
//}

template<class T, class Derived>
array_base<T, Derived> :: ~array_base()
{
    free(array_t);
    free(array);
}


/*
 * ****************************************************************************
 * ****************************** Operators ***********************************
 * ****************************************************************************
 */


#ifndef PERIODIC
template <class T, class Derived>
T array_base<T, Derived> :: operator()(const uint t, const uint n, const uint m) const
{
    if(!bounds(t, n, m))
        throw out_of_bounds_err(string("T array_base<T, Derived> :: operator()(uint, uint, uint): out of bounds\n"));
    return (*(array_t[t] + address(n,m)));
}


template <class T, class Derived>
T& array_base<T, Derived> :: operator()(const uint t, const uint n, const uint m)
{
    if(!bounds(t, n, m))
        throw out_of_bounds_err(string("T& array_base<T> :: operator()(uint, uint, uint): out of bounds\n"));
    return (*(array_t[t] + address(n,m)));
}


template <class T, class Derived>
T array_base<T, Derived> :: operator()(const uint n, const uint m) const
{
    if(!bounds(n, m))
        throw out_of_bounds_err(string("T array_base<T> :: operator()(uint, uint, uint): out of bounds\n"));
    return (*(array + address(n,m)));
}


template <class T, class Derived>
T& array_base<T, Derived> :: operator()(const uint n, const uint m)
{
    if(!bounds(n, m))
        throw out_of_bounds_err(string("T& array_base<T> :: operator()(uint, uint, uint): out of bounds\n"));
    return (*(array + address(n,m)));
}
#endif //PERIODIC


#ifdef PERIODIC
template <class T, class Derived>
T array_base<T, Derived> :: operator()(const uint t, int n_in, int m_in) const
{
    uint n = (n_in > 0 ? n_in : Nx + n_in) % Nx;
    uint m = (m_in > 0 ? m_in : My + m_in) % My;
    if(n_in < 0)
        cout << "n_in = " << n_in << "-> " << n << "\n";
    if(!bounds(t, n, m))
        throw out_of_bounds_err(string("T array_base<T, Derived> :: operator()(uint, uint, uint): out of bounds\n"));
    return (*(array_t[t] + address(n,m)));
}


template <class T, class Derived>
T& array_base<T, Derived> :: operator()(const uint t, int n_in, int m_in)
{
    uint n = (n_in > 0 ? n_in : Nx + n_in) % Nx;
    uint m = (m_in > 0 ? m_in : My + m_in) % My;
    if(n_in < 0)
        cout << "n_in = " << n_in << "-> " << n << "\n";
    if(!bounds(t, n, m))
        throw out_of_bounds_err(string("T& array_base<T> :: operator()(uint, uint, uint): out of bounds\n"));
    return (*(array_t[t] + address(n,m)));
}


template <class T, class Derived>
T array_base<T, Derived> :: operator()(int n_in, int m_in) const
{
    uint n = (n_in > 0 ? n_in : Nx + n_in) % Nx;
    uint m = (m_in > 0 ? m_in : My + m_in) % My;
    if(n_in < 0)
        cout << "n_in = " << n_in << "-> " << n << "\n";
    if(!bounds(n, m))
        throw out_of_bounds_err(string("T array_base<T> :: operator()(uint, uint, uint): out of bounds\n"));
    return (*(array + address(n,m)));
}


template <class T, class Derived>
T& array_base<T, Derived> :: operator()(int n_in, int m_in)
{
    uint n = (n_in > 0 ? n_in : Nx + n_in) % Nx;
    uint m = (m_in > 0 ? m_in : My + m_in) % My;
    if(n_in < 0)
        cout << "n_in = " << n_in << "-> " << n << "\n";
    if(!bounds(n, m))
        throw out_of_bounds_err(string("T& array_base<T> :: operator()(uint, uint, uint): out of bounds\n"));
    return (*(array + address(n,m)));
}

#endif // PERIODIC

template <class T, class Derived>
Derived& array_base<T, Derived> :: operator=(const T& rhs)
{
    uint n{0}, m{0};
    for(n = 0; n < Nx; n++){
        for(m = 0; m < My; m++){
            (*this)(0, n, m) = rhs;
        }
    }
    return static_cast<Derived &>(*this);
}


template <class T, class Derived>
Derived& array_base<T, Derived> :: operator=(const array_base<T, Derived>& rhs)
{
    //cout << "array_base<T>& array_base<T> :: operator=(const array_base<T>& rhs)\n";
    // Sanity checks
    if(!bounds(rhs.get_tlevs(), rhs.get_nx(), rhs.get_my()))
        throw out_of_bounds_err(string("array_base<T>& array_base<T>::operator=(const array_base<T>& rhs): Index out of bounds: n\n"));
    if( (void*) this == (void*) &rhs)
        return static_cast<Derived &>(*this);

    memcpy(array, rhs.get_array(0), sizeof(T) * Nx * My);
    return static_cast<Derived &>(*this);
}


// C++ 11 feature. Inactive but tested
//template <class T, class Derived>
//array_base<T>& array_base<T>::operator=(array_base<T>&& rhs)
//{
//    //cout << "array_base<T>& array_base<T> :: operator=(const array_base<T>&& rhs)\n";
//    // Sanity checks
//    if(!bounds(rhs.get_tlevs(), rhs.get_nx(), rhs.get_my()))
//        throw out_of_bounds_err(string("array_base<T>& array_base<T>::operator=(const array_base<T>& rhs): Index out of bounds: n\n"));
//    if( (void*) this == (void*) &rhs)
//        return *this;
//
//    T* tmp_array = array;
//    T** tmp_array_t = array_t;
//    array = rhs.array;
//    array_t = rhs.array_t;
//    rhs.array = tmp_array;
//    rhs.array_t = tmp_array_t;
//
//    return *this;
//}



template <class T, class Derived>
Derived& array_base<T, Derived> :: operator+=(const Derived& rhs)
{
    //cout << "array_base<T>& array_base<T> :: operator+=(const array_base<T>& rhs)\n";
    if(!bounds(rhs.get_tlevs(), rhs.get_nx(), rhs.get_my()))
        throw out_of_bounds_err(string("fftw_array::operator=(const fftw_array& rhs): Index out of bounds: n\n"));
	if ( (void*) this == (void*) &rhs) 
        throw operator_err(string("array_base<T>& array_base<T>::operator*=(const array_base<T>& rhs): this == rhs\n"));

    std::vector<std::thread> thr;
    for(unsigned int n = 0; n < nthreads; n++) 
        thr.push_back(std::thread(calc_loop<Derived, std::plus<T>>, static_cast<Derived*>(this), &rhs, n * nelem, (n + 1) * nelem));

    for(auto &t : thr)
        t.join();
    
	return static_cast<Derived &>(*this);
}


template <class T, class Derived>
Derived& array_base<T, Derived> :: operator-=(const Derived& rhs)
{
    if(!bounds(rhs.get_tlevs(), rhs.get_nx(), rhs.get_my()))
        throw out_of_bounds_err(string("fftw_array::operator=(const fftw_array& rhs): Index out of bounds: n\n"));
	if ( (void*) this == (void*) &rhs) 
        throw operator_err(string("array_base<T>& array_base<T>::operator-=(const array_base<T>& rhs): this == rhs\n"));

    std::vector<std::thread> thr;
    for(unsigned int n = 0; n < nthreads; n++) 
        //thr.push_back(std::thread(calc_loop<array_base<T>, std::minus<T>>,this, &rhs, n * nelem, (n + 1) * nelem));
        thr.push_back(std::thread(calc_loop<Derived, std::minus<T>>, static_cast<Derived *>(this), &rhs, n * nelem, (n + 1) * nelem));

    for(auto &t : thr)
        t.join();
	return static_cast<Derived &>(*this);
}


template <class T, class Derived>
Derived& array_base<T, Derived> :: operator*=(const Derived& rhs)
{
    if(!bounds(rhs.get_tlevs(), rhs.get_nx(), rhs.get_my()))
        throw out_of_bounds_err(string("array_base::operator=(const fftw_array& rhs): Index out of bounds: n\n"));
	if ( (void*) this == (void*) &rhs) 
        throw operator_err(string("array_base<T>& array_base<T>::operator*=(const array_base<T>& rhs): this == rhs\n"));

    std::vector<std::thread> thr;
    for(unsigned int n = 0; n < nthreads; n++) 
        thr.push_back(std::thread(calc_loop<Derived, std::multiplies<T>>, static_cast<Derived *>(this), &rhs, n * nelem, (n + 1) * nelem));

    for(auto &t : thr)
        t.join();

	return static_cast<Derived &>(*this);
}


template <class T, class Derived>
Derived& array_base<T, Derived> :: operator+=(const T& rhs)
{
    std::vector<std::thread> thr;
    for(unsigned int n = 0; n < nthreads; n++) 
        thr.push_back(std::thread(calc_loop_scalar<Derived, T, std::plus<T>>, static_cast<Derived *>(this), rhs, n * nelem, (n + 1) * nelem));

    for(auto &t : thr)
        t.join();
	return static_cast<Derived &>(*this);
}

template <class T, class Derived>
Derived& array_base<T, Derived> :: operator-=(const T& rhs)
{
    //for(unsigned int n = 0; n < nthreads; n++) 
    //    calc_loop_scalar<array_base<T>, T, std::minus<T>>(this, rhs, n * nelem, (n + 1) * nelem - 1);
    //calc_loop_scalar<array_base<T>, T, std::minus<T>>(this, rhs); 
    std::vector<std::thread> thr;
    for(unsigned int n = 0; n < nthreads; n++) 
        thr.push_back(std::thread(calc_loop_scalar<Derived, T, std::minus<T>>, static_cast<Derived *>(this), rhs, n * nelem, (n + 1) * nelem));

    for(auto &t : thr)
        t.join();
	return static_cast<Derived &>(*this);
}

template <class T, class Derived>
Derived& array_base<T, Derived> :: operator*=(const T& rhs)
{
    //for(unsigned int n = 0; n < nthreads; n++) 
    //    calc_loop_scalar<array_base<T>, T, std::multiplies<T>>(this, rhs, n * nelem, (n + 1) * nelem - 1);
    //calc_loop_scalar<array_base<T>, T, std::multiplies<T>>(this, rhs); 
    std::vector<std::thread> thr;
    for(unsigned int n = 0; n < nthreads; n++) 
        thr.push_back(std::thread(calc_loop_scalar<Derived, T, std::multiplies<T>>, static_cast<Derived *>(this), rhs, n * nelem, (n + 1) * nelem));

    for(auto &t : thr)
        t.join();
	return static_cast<Derived &>(*this);
}


template <class T, class Derived>
Derived array_base<T, Derived>::operator+(const Derived& rhs) const 
{
    //array_base<T> result(*this);
    // Cast away const-ness, constructor does not hurt *this :)
    Derived result(const_cast<array_base<T, Derived>*>(this));
    result += rhs;
    return(result);
}


template <class T, class Derived>
Derived array_base<T, Derived>::operator-(const Derived& rhs) const 
{
    //array_base<T> result(*this);
    Derived result(const_cast<array_base<T, Derived>*>(this));
    result -= rhs;
    return(result);
}


template <class T, class Derived>
Derived array_base<T, Derived>::operator*(const T& rhs) const 
{
	//array_base<T> result(*this);
    Derived result(const_cast<array_base<T, Derived>*>(this));
	result *= rhs;
	return(result);
}


template <class T, class Derived>
Derived array_base<T, Derived>::operator*(const Derived& rhs) const
{
    //array_base<T> result(*this);
    Derived result(const_cast<array_base<T, Derived>*>(this));
    result *= rhs;
    return result;
}


// Copy memory from t_src to t_dst
template <class T, class Derived>
void array_base<T, Derived>::copy(const uint t_dst, const uint t_src)
{
    if(!bounds(t_src, 0, 0) || !bounds(t_dst, 0, 0))
        throw out_of_bounds_err("void array_base<T>::copy_array(uint dst , uint src): out of bounds\n");
    memcpy(array_t[t_dst], array_t[t_src], sizeof(T) * Nx * My);
}


// Copy memory from another array to this array
template <class T, class Derived>
void array_base<T, Derived>::copy(const uint t_dst, const Derived& src, const uint t_src)
{
    if(!bounds(t_src, 0, 0) || !bounds(t_dst, 0, 0))
        throw out_of_bounds_err("void array_base<T>::copy_array(uint t_dst, array_base<T> src, uint t_src): out of bounds\n");
    memcpy(array_t[t_dst], src.array_t[t_src], sizeof(T) * Nx * My);
}

// Move t_src into t_dst, zero out t_src
template <class T, class Derived>
void array_base<T, Derived>::move(const uint t_dst, const uint t_src)
{
    if (t_dst < tlevs-1 && t_src < tlevs-1)
    {
        T* tmp = array_t[t_dst];
        array_t[t_dst] = array_t[t_src];
        array_t[t_src] = tmp;
        memset(array_t[t_src], 0, Nx * My * sizeof(T));
    } 
    else
    {
        throw out_of_bounds_err("void array_base<T>::move_array(uint t_dst, uint t_src): out of bounds\n");
    }
}


template <class T, class Derived>
void array_base<T, Derived>::advance()
{
    uint t;
    //for(t = tlevs - 1; t > 0; t--)
    //    cout << "array[" << t << "] at " << array_t[t] << "\n";

    T* tmp = array_t[tlevs-1];

    for (unsigned int t = tlevs - 1; t > 0; t--)
        array_t[t] = array_t[t - 1];
    
    array_t[0] = tmp;
    // Set region array_t[0] points to to zero
    memset(array_t[0], 0, sizeof(T) * Nx * My);
}

#endif//ARRAY_BASE_H
// End of file array_base.h
