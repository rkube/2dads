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

/// @brief Template for the loop operator used in operator[+-*]=, run by each thread
///
/// @detailed Here R, i.e. array_base<double>, array_base<my_cmplex>,
/// @detailed Functor is a function that takes two arguments of the base type of R,
/// @detailed i.e. double, complex and returns a reference to the operators,
/// @detailed see std::plus<T>, std::minus<T>, std::multiplies<T>, etc
/// @param lhs LHS of operation
/// @param rhs RHS of operation
/// @param n_start start of n index for loop of executing thread
/// @param n_end end of n index for loop of executing thread
template<class R, typename Functor>
#ifdef PERIODIC
inline void calc_loop(R* lhs, const R* rhs, int n_start, int n_end) 
#endif
#ifndef PERIODIC
inline void calc_loop(R* lhs, const R* rhs, unsigned int n_start, unsigned int n_end) 
#endif
{
    Functor op;
#ifdef PERIODIC
    int n{n_start};
    int m{0};
    const int My{int(rhs -> get_my())};
#endif
#ifndef PERIODIC
    unsigned int n{n_start};
    unsigned int m{0};
    const uint My{rhs -> get_my()};
#endif

    for(n = n_start; n < n_end; n++)
        for(m = 0; m < My; m++)
            (*lhs)(0, n, m) =  op( (*lhs)(0, n, m), (*rhs)(0, n, m));
}

/// @breif Same as calc_loop, but rhs is a scalar
/// @param lhs Left hand side of operator
/// @param rhs Right hand side of operator, scalar
/// @param n_start start of n index for loop of executing thread
/// @param n_end end of n index for loop of executing thread
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
    int n{n_start};
    int m{0};
    const int My{int(rhs -> get_my())};
#endif
#ifndef PERIODIC
    unsigned int n{n_start};
    unsigned int m{0};
    const uint My{lhs -> get_my()};
#endif    
    for(n = n_start; n < n_end; n++)
        for(m = 0; m < My; m++)
            (*lhs)(0, n, m) = op( (*lhs)(0, n, m), rhs);
}

/// @brief Base class for threaded multi-dimensional array
/// @details Template parameters:
/// @details T: template type of the derived class, i.e. double, complex, int, etc
/// @details Derived: type of the derived class. We use this, so that the return type of the 
/// @details operators defined in this class return types of the derived class calling the methods,
/// @details and no array_base.
///
/// @details See https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern

template <class T, class Derived> 
class array_base{
    public:
        using uint = unsigned int;
        array_base() = delete;
        // Copy constructor
        array_base(const array_base<T, Derived>&); ///<Copy constructor 
        array_base(array_base<T, Derived>&&); ///<Move constructor 
        array_base(array_base<T, Derived>*); ///< Copy constructor
        // Move constructor
        array_base(const uint, const uint, const uint, const uint); ///< Specify number of threads, tlevs, Nx, My
        array_base(const uint, const uint, const uint); ///< Create with tlevs = 1
        array_base(const uint, const uint); ///< Create with nthreads = 1, tlevs = 1
        ~array_base();

        // Set number of threads
        inline void set_numthreads(uint);

        // Operators
        // use -DPERIODIC, to wrap indices on array boundaries, i.e. n = -1 -> Nx - 1, etc...
#ifdef PERIODIC
        inline T& operator()(const uint, int, int); ///< Write-access with boundary wrapping for spatial indices
        inline T operator() (const uint, int, int) const; ///< Read access with boundary wrapping for spatial indices

        inline T& operator()(int, int);  ///< Write-access with boundary wrapping for spatial indices
        inline T operator() (int, int) const;  ///< Read access with boundary wrapping for spatial indices
#endif // PERIODIC
        // Use const uint as input since nx... is not modified in operator() implementation
#ifndef PERIODOC
        inline T& operator()(const uint, const uint, const uint); ///< Write access to elements, t=0
        inline T operator() (const uint, const uint, const uint) const; ///< Write access to elements, t=0

        inline T& operator()(const uint, const uint); ///< Write access to elements
        inline T operator() (const uint, const uint) const; ///< Read-access to elements
#endif

        Derived& operator=(const T&); ///< Set elements for tlev=1
        Derived& operator=(const array_base<T, Derived>&); ///< Copy to previously allocated array
        // C++11
        array_base<T, Derived>& operator=(array_base<T, Derived>&&); ///< Copy result of temporary computation

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
        void advance(); ///< Advance time levels 0->1, 1->2, ... tlev -> 0 (zeroed out)
        void copy(const uint t_dst, const uint t_src); ///< copy memory pointed to by array_t[t_src] to array_t[t_dst]
        void copy(const uint, const Derived&, const uint); ///< Copy from another array
        void move(const uint, const uint); ///< set array_t pointer from t_src to t_dst, zero out t_src

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
        inline uint get_tlevs() const {return tlevs;}
        inline uint get_nx() const {return Nx;}
        inline uint get_my() const {return My;}
        inline uint get_nelem() const{return nelem;}
        inline uint get_nthreads() const {return nthreads;}
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

/// @brief Standard constructor for class array_base<T>, initialize arrar bounds, allocate memory and initialize threading data
/// @param nthr number of threads
/// @param t number of time levels
/// @param n Nx
/// @param m My
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



/// @brief Create array with nthreads = 1, tlevs = 1
/// @param n Nx
/// @param m My
template<class T, class Derived>
array_base<T, Derived> :: array_base(uint n, uint m) :
    array_base(1, 1, n, m)
{
}


/// @brief Copy constructor 
/// @details Call base constructor and copies T* array from src to dst, all time levels
/// @param src Source
template<class T, class Derived>
array_base<T, Derived> :: array_base(const array_base<T, Derived>& src) :
    array_base(src.nthreads, src.tlevs, src.Nx, src.My)
{
    memcpy(array, src.array, sizeof(T) * tlevs * Nx * My);
}


/// @brief Copy constructor 
/// @details Calls base constructor array_base(uint, uint, uint, uint) with
/// @details values from src
/// @param src Source
template<class T, class Derived>
array_base<T, Derived> :: array_base(array_base<T, Derived>* src) :
    array_base((*src).nthreads, (*src).tlevs, (*src).Nx, (*src).My)
{
    memcpy(array, (*src).array, sizeof(T) * tlevs * Nx * My);
}



/// @brief Move constructor
/// @details Takes a rvalue as a reference and moves the data of this object into its own private variables.
/// @details This is called in f.ex.:
/// @details array_base<twodads::real_t> a1(t, N, M);
/// @details array_base<twodads::real_t> a2(t, N, M);
/// @details a1 = 1.0;
/// @details a2 = 2.0;
/// @details array_base<twodads::real_t> a3(a1 + a2);
/// @details The constructor just copies the data of the temporary object "a1 + a2" 
///
/// @details src is not const in this case
/// @details Implement swap semantics for points to array and array_t between
/// @details callar and RHS
/// @param rhs Source

template<class T, class Derived>
array_base<T, Derived> :: array_base(array_base<T, Derived>&& rhs) :
    nthreads(rhs.nthreads),
    nelem(rhs.nelem),
    tlevs(rhs.tlevs),
    Nx(rhs.Nx), 
    My(rhs.My),
    bounds(tlevs, Nx, My)
{
    T* tmp_array = array;
    T** tmp_array_t = array_t;
    array = rhs.array;
    array_t = rhs.array_t;
    rhs.array = tmp_array;
    rhs.array_t = tmp_array_t;
}


/// @brief free memory pointed to by array and array_t
template<class T, class Derived>
array_base<T, Derived> :: ~array_base()
{
    free(array_t);
    free(array);
}


/// @brief Set number of threads
template<class T, class Derived>
void array_base<T, Derived> :: set_numthreads(unsigned int nthr)
{
    assert(Nx % nthreads == 0);
    nthreads = nthr;
    nelem = Nx / nthreads;
}


/*
 * ****************************************************************************
 * ****************************** Operators ***********************************
 * ****************************************************************************
 */


#ifndef PERIODIC
template <class T, class Derived>
inline T array_base<T, Derived> :: operator()(const uint t, const uint n, const uint m) const
{
#ifdef DEBUG
    if(!bounds(t, n, m))
        throw out_of_bounds_err(string("T array_base<T, Derived> :: operator()(uint, uint, uint): out of bounds\n"));
#endif
    return (*(array_t[t] + address(n,m)));
}


template <class T, class Derived>
inline T& array_base<T, Derived> :: operator()(const uint t, const uint n, const uint m)
{
#ifdef DEBUG
    if(!bounds(t, n, m))
        throw out_of_bounds_err(string("T& array_base<T> :: operator()(uint, uint, uint): out of bounds\n"));
#endif
    return (*(array_t[t] + address(n,m)));
}


template <class T, class Derived>
inline T array_base<T, Derived> :: operator()(const uint n, const uint m) const
{
#ifdef DEBUG
    if(!bounds(n, m))
        throw out_of_bounds_err(string("T array_base<T> :: operator()(uint, uint, uint): out of bounds\n"));
#endif
    return (*(array + address(n,m)));
}


template <class T, class Derived>
inline T& array_base<T, Derived> :: operator()(const uint n, const uint m)
{
#ifdef DEBUG
    if(!bounds(n, m))
        throw out_of_bounds_err(string("T& array_base<T> :: operator()(uint, uint, uint): out of bounds\n"));
#endif 
    return (*(array + address(n,m)));
}
#endif //PERIODIC


#ifdef PERIODIC
template <class T, class Derived>
inline T array_base<T, Derived> :: operator()(const uint t, int n_in, int m_in) const
{
    uint n = (n_in > 0 ? n_in : Nx + n_in) % Nx;
    uint m = (m_in > 0 ? m_in : My + m_in) % My;
    //if(n_in < 0)
    //    cout << "n_in = " << n_in << "-> " << n << "\n";
#ifdef DEBUG
    if(!bounds(t, n, m))
        throw out_of_bounds_err(string("T array_base<T, Derived> :: operator()(uint, uint, uint): out of bounds\n"));
#endif
    return (*(array_t[t] + address(n,m)));
}


template <class T, class Derived>
inline T& array_base<T, Derived> :: operator()(const uint t, int n_in, int m_in)
{
    uint n = (n_in > 0 ? n_in : Nx + n_in) % Nx;
    uint m = (m_in > 0 ? m_in : My + m_in) % My;
    //if(n_in < 0)
    //    cout << "n_in = " << n_in << "-> " << n << "\n";
#ifdef DEBUG
    if(!bounds(t, n, m))
        throw out_of_bounds_err(string("T& array_base<T> :: operator()(uint, uint, uint): out of bounds\n"));
#endif
    return (*(array_t[t] + address(n,m)));
}


template <class T, class Derived>
inline T array_base<T, Derived> :: operator()(int n_in, int m_in) const
{
    uint n = (n_in > 0 ? n_in : Nx + n_in) % Nx;
    uint m = (m_in > 0 ? m_in : My + m_in) % My;
    //if(n_in < 0)
    //    cout << "n_in = " << n_in << "-> " << n << "\n";
#ifdef DEBUG
    if(!bounds(n, m))
        throw out_of_bounds_err(string("T array_base<T> :: operator()(uint, uint, uint): out of bounds\n"));
#endif
    return (*(array + address(n,m)));
}


template <class T, class Derived>
inline T& array_base<T, Derived> :: operator()(int n_in, int m_in)
{
    uint n = (n_in > 0 ? n_in : Nx + n_in) % Nx;
    uint m = (m_in > 0 ? m_in : My + m_in) % My;
    //if(n_in < 0)
    //    cout << "n_in = " << n_in << "-> " << n << "\n";
#ifdef DEBUG
    if(!bounds(n, m))
        throw out_of_bounds_err(string("T& array_base<T> :: operator()(uint, uint, uint): out of bounds\n"));
#endif
    return (*(array + address(n,m)));
}

#endif // PERIODIC

/// @brief set tlev=0 to rhs
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


/// @brief Copy memory pointed to by rhs.array_t[0] to array_t[0]
template <class T, class Derived>
Derived& array_base<T, Derived> :: operator=(const array_base<T, Derived>& rhs)
{
    if(!bounds(rhs.get_tlevs(), rhs.get_nx(), rhs.get_my()))
        throw out_of_bounds_err(string("array_base<T>& array_base<T>::operator=(const array_base<T>& rhs): Index out of bounds: n\n"));
    if( (void*) this == (void*) &rhs)
        return static_cast<Derived &>(*this);

    memcpy(array, rhs.get_array(0), sizeof(T) * Nx * My);
    return static_cast<Derived &>(*this);
}


/// @brief Move result of temporary object to LHS
/// @detailed Swap array and array_t of this with rhs
template <class T, class Derived>
array_base<T, Derived>& array_base<T, Derived>::operator=(array_base<T, Derived>&& rhs)
{
    if(!bounds(rhs.get_tlevs(), rhs.get_nx(), rhs.get_my()))
        throw out_of_bounds_err(string("array_base<T>& array_base<T>::operator=(const array_base<T>& rhs): Index out of bounds: n\n"));
    if( (void*) this == (void*) &rhs)
        return *this;

    T* tmp_array = array;
    T** tmp_array_t = array_t;
    array = rhs.array;
    array_t = rhs.array_t;
    rhs.array = tmp_array;
    rhs.array_t = tmp_array_t;

    return *this;
}


/// @brief Add rhs to this, uses threads
template <class T, class Derived>
Derived& array_base<T, Derived> :: operator+=(const Derived& rhs)
{
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


/// @brief Subtract rhs from this, tlev=0 for both. Uses threads
template <class T, class Derived>
Derived& array_base<T, Derived> :: operator-=(const Derived& rhs)
{
    if(!bounds(rhs.get_tlevs(), rhs.get_nx(), rhs.get_my()))
        throw out_of_bounds_err(string("fftw_array::operator=(const fftw_array& rhs): Index out of bounds: n\n"));
	if ( (void*) this == (void*) &rhs) 
        throw operator_err(string("array_base<T>& array_base<T>::operator-=(const array_base<T>& rhs): this == rhs\n"));

    std::vector<std::thread> thr;
    for(unsigned int n = 0; n < nthreads; n++) 
        thr.push_back(std::thread(calc_loop<Derived, std::minus<T>>, static_cast<Derived *>(this), &rhs, n * nelem, (n + 1) * nelem));

    for(auto &t : thr)
        t.join();
	return static_cast<Derived &>(*this);
}


/// @brief Multiple this with rhs, tlev=0 for both. Uses threads
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


/// @brief Add rhs to this, tlev = 0. Uses threads
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


/// @brief Subtract rhs from this, tlev = 0. Uses threads
template <class T, class Derived>
Derived& array_base<T, Derived> :: operator-=(const T& rhs)
{
    std::vector<std::thread> thr;
    for(unsigned int n = 0; n < nthreads; n++) 
        thr.push_back(std::thread(calc_loop_scalar<Derived, T, std::minus<T>>, static_cast<Derived *>(this), rhs, n * nelem, (n + 1) * nelem));

    for(auto &t : thr)
        t.join();
	return static_cast<Derived &>(*this);
}


/// @brief Multiply this with rhs, tle = 0. Uses threads.
template <class T, class Derived>
Derived& array_base<T, Derived> :: operator*=(const T& rhs)
{
    std::vector<std::thread> thr;
    for(unsigned int n = 0; n < nthreads; n++) 
        thr.push_back(std::thread(calc_loop_scalar<Derived, T, std::multiplies<T>>, static_cast<Derived *>(this), rhs, n * nelem, (n + 1) * nelem));

    for(auto &t : thr)
        t.join();
	return static_cast<Derived &>(*this);
}


/// @brief add this and rhs via operator+=
template <class T, class Derived>
Derived array_base<T, Derived>::operator+(const Derived& rhs) const 
{
    // Cast away const-ness, constructor does not hurt *this :)
    Derived result(const_cast<array_base<T, Derived>*>(this));
    result += rhs;
    return(result);
}


/// @brief Subtract rhs from this via operator-=
template <class T, class Derived>
Derived array_base<T, Derived>::operator-(const Derived& rhs) const 
{
    Derived result(const_cast<array_base<T, Derived>*>(this));
    result -= rhs;
    return(result);
}


/// @brief Multiply this and rhs via operator *=
template <class T, class Derived>
Derived array_base<T, Derived>::operator*(const T& rhs) const 
{
    Derived result(const_cast<array_base<T, Derived>*>(this));
	result *= rhs;
	return(result);
}


/// @brief Multiply this and rhs via operator *=
template <class T, class Derived>
Derived array_base<T, Derived>::operator*(const Derived& rhs) const
{
    Derived result(const_cast<array_base<T, Derived>*>(this));
    result *= rhs;
    return result;
}


/// @brief Copy memory from t_src to t_dst
template <class T, class Derived>
void array_base<T, Derived>::copy(const uint t_dst, const uint t_src)
{
    if(!bounds(t_src, 0, 0) || !bounds(t_dst, 0, 0))
        throw out_of_bounds_err("void array_base<T>::copy_array(uint dst , uint src): out of bounds\n");
    memcpy(array_t[t_dst], array_t[t_src], sizeof(T) * Nx * My);
}


/// @brief Copy memory from another array to this array
/// @param t_dst time level array_t[t_dst]
/// @param src Source for copy operation
/// @param t_src Time level to be copied from src
template <class T, class Derived>
void array_base<T, Derived>::copy(const uint t_dst, const Derived& src, const uint t_src)
{
    if(!bounds(t_src, 0, 0) || !bounds(t_dst, 0, 0))
        throw out_of_bounds_err("void array_base<T>::copy_array(uint t_dst, array_base<T> src, uint t_src): out of bounds\n");
    memcpy(array_t[t_dst], src.array_t[t_src], sizeof(T) * Nx * My);
}


/// @brief Move data from t_src to t_dst, zero out time level t_src
/// @param t_src Source time level
/// @param t_dst Destination time level
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


/// @brief push data from time levels t -> t + 1
/// @detailed Zero out t=0
template <class T, class Derived>
void array_base<T, Derived>::advance()
{
    uint t;

    T* tmp = array_t[tlevs-1];

    for (unsigned int t = tlevs - 1; t > 0; t--)
        array_t[t] = array_t[t - 1];
    
    array_t[0] = tmp;
    // Set region array_t[0] points to to zero
    memset(array_t[0], 0, sizeof(T) * Nx * My);
}

#endif//ARRAY_BASE_H
// End of file array_base.h
