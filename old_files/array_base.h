/*!
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
#include "bounds.h"
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
inline void calc_loop(R* lhs, const R* rhs, int m_start, int m_end) 
#endif
#ifndef PERIODIC
inline void calc_loop(R* lhs, const R* rhs, unsigned int m_start, unsigned int m_end) 
#endif
{
    Functor op;
#ifdef PERIODIC
    int m{m_start};
    int n{0};
    const int Nx{int(rhs -> get_nx())};
#endif
#ifndef PERIODIC
    unsigned int m{m_start};
    unsigned int n{0};
    const uint Nx{rhs -> get_nx()};
#endif

    // Operate on (m_end - m_start) consecutive rows from memory
    for(m = m_start; m < m_end; m++)
        for(n = 0; n < Nx; n++)
            (*lhs)(0, m, n) =  op( (*lhs)(0, m, n), (*rhs)(0, m, n));
}

/// @breif Same as calc_loop, but rhs is a scalar
/// @param lhs Left hand side of operator
/// @param rhs Right hand side of operator, scalar
/// @param n_start start of n index for loop of executing thread
/// @param n_end end of n index for loop of executing thread
template<class R, class S, typename Functor>
#ifdef PERIODIC
inline void calc_loop_scalar(R* lhs, const S& rhs, int m_start, int m_end)
#endif
#ifndef PERIODIC
inline void calc_loop_scalar(R* lhs, const S& rhs, unsigned int m_start, unsigned int m_end)
#endif
{
    Functor op;

#ifdef PERIODIC
    int m{m_start};
    int n{0};
    const int Nx{int(rhs -> get_nx())};
#endif
#ifndef PERIODIC
    unsigned int m{m_start};
    unsigned int n{0};
    const uint Nx{lhs -> get_nx()};
#endif    
    for(m = m_start; m < m_end; m++)
        for(n = 0; n < Nx; n++)
            (*lhs)(0, m, n) = op( (*lhs)(0, m, n), rhs);
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
        array_base(const array_base<T, Derived>*); ///< Copy constructor
        // Move constructor
        array_base(const uint, const uint, const uint, const uint); ///< Specify number of threads, tlevs, My, Nx
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
        inline T& operator()(uint, uint, uint); ///< Write access to elements, t=0
        inline T operator() (uint, uint, uint) const; ///< Write access to elements, t=0

        inline T& operator()(uint, uint); ///< Write access to elements
        inline T operator() (uint, uint) const; ///< Read-access to elements
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

            for(uint t = 0; t < tl; t++)
            {
                os << "t: " << t << "\n";
                for(uint m = 0; m < my; m++)
                {
                    for(uint n = 0; n < nx; n++)
                    {
                        os << src(t, m, n) << "\t";
                    }
                    os << "\n";
                }
            }
            return (os);
        }


        // Bounds function
        // Return the array element specified by (n,m)
        // time indexing is handled only via array_t
        inline int address(const uint m, const uint n) const {return(m * Nx + n);}

        // Access to dimensions
        inline uint get_tlevs() const {return tlevs;}
        inline uint get_my() const {return My;}
        inline uint get_nx() const {return Nx;}
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
        const uint My;
        const uint Nx;
        // Check for bounds
        bounds check_bounds;
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
array_base<T, Derived> :: array_base(const uint nthr, const uint t, const uint m, const uint n) :
    nthreads(nthr), 
    tlevs(t),
    My(m),
    Nx(n),
    check_bounds(tlevs, My, Nx),
    array(nullptr),
    array_t(nullptr)
{
    // Allocate memory
    array = new T[tlevs * My * Nx];
    array_t = new T*[tlevs];
    for(uint tl = 0; tl < tlevs; tl++)
        array_t[tl] = &array[tl * My * Nx];
    
    // Only allow a number of threads that divides 
    // Nx. If not, abort.
    assert(My % nthreads == 0);
    nelem = My / nthreads;

//#ifdef DEBUG
//    cout << "array_base<T>::array_base(uint, uint, uint)\n";
//    cout << "\t\ttlevs = " << tlevs << "\tMy = " << My << "\tNx = " << Nx << "\n";
//    cout << "\t\tarray: " << sizeof(T) * tlevs * My * Nx << " bytes at " << array << "\n";
//    cout << "\t\tarray_t: " << sizeof(T*) * tlevs << " bytes at " << array_t << "\n";
//    for(uint tl = 0; tl < tlevs; tl++)
//        cout << "\t\tarray_t[" << tl << "] at " << array_t[tl] << "\n";
//    cout << "\t\tnthreads = " << nthreads << " threads, nelem = " << nelem << "\n";
//#endif
}



/// @brief Create array with nthreads = 1, tlevs = 1
/// @param m My
/// @param n Nx
template<class T, class Derived>
array_base<T, Derived> :: array_base(const uint m, const uint n) :
    array_base(1, 1, m, n)
{
}


/// @brief Copy constructor 
/// @details Call base constructor and copies T* array from src to dst, all time levels
/// @param src Source
template<class T, class Derived>
array_base<T, Derived> :: array_base(const array_base<T, Derived>& src) :
    array_base(src.nthreads, src.tlevs, src.My, src.Nx)
{
    memcpy(array, src.array, sizeof(T) * tlevs * My * Nx);
}


/// @brief Copy constructor 
/// @details Calls base constructor array_base(uint, uint, uint, uint) with
/// @details values from src
/// @param src Source
template<class T, class Derived>
array_base<T, Derived> :: array_base(const array_base<T, Derived>* src) :
    array_base((*src).nthreads, (*src).tlevs, (*src).My, (*src).Nx)
{
    memcpy(array, (*src).array, sizeof(T) * tlevs * My * Nx);
}



/// @brief Move constructor
/// @details Takes a rvalue as a reference and moves the data of this object into its own private variables.
/// @details This is called in f.ex.:
/// @details array_base<twodads::real_t> a1(t, M, N);
/// @details array_base<twodads::real_t> a2(t, M, N);
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
    My(rhs.My),
    Nx(rhs.Nx), 
    check_bounds(tlevs, My, Nx)
{
    // See Stroustrup C++, 17.5.2
    array = rhs.array;
    array_t = rhs.array_t;
    rhs.array = nullptr;
    rhs.array_t = nullptr;
}


/// @brief free memory pointed to by array and array_t
template<class T, class Derived>
array_base<T, Derived> :: ~array_base()
{
    delete[] array;
    delete[] array_t;
}


/// @brief Set number of threads
template<class T, class Derived>
void array_base<T, Derived> :: set_numthreads(unsigned int nthr)
{
    assert(My % nthreads == 0);
    nthreads = nthr;
    nelem = My / nthreads;
}


/*
 * ****************************************************************************
 * ****************************** Operators ***********************************
 * ****************************************************************************
 */


#ifndef PERIODIC
template <class T, class Derived>
inline T& array_base<T, Derived> :: operator()(uint t, uint m, uint n)
{
#ifdef DEBUG
	check_bounds(t, m, n);
#endif
    return (*(array_t[t] + address(m, n)));
}


template <class T, class Derived>
inline T array_base<T, Derived> :: operator()(uint t, uint m, uint n) const
{
#ifdef DEBUG
	check_bounds(t, m, n);
#endif
    return (*(array_t[t] + address(m, n)));
}


template <class T, class Derived>
inline T& array_base<T, Derived> :: operator()(uint m, uint n)
{
#ifdef DEBUG
	check_bounds(m, n);
#endif 
    return (*(array + address(m, n)));
}


template <class T, class Derived>
inline T array_base<T, Derived> :: operator()(uint m, uint n) const
{
#ifdef DEBUG
	check_bounds(m, n);
#endif
    return (*(array + address(m, n)));
}
#endif //PERIODIC


#ifdef PERIODIC
template <class T, class Derived>
inline T array_base<T, Derived> :: operator()(const uint t, int m_in, int n_in) const
{
    uint m = (m_in > 0 ? m_in : My + m_in) % My;
    uint n = (n_in > 0 ? n_in : Nx + n_in) % Nx;
    //if(n_in < 0)
    //    cout << "n_in = " << n_in << "-> " << n << "\n";
#ifdef DEBUG
    check_bounds(m, n);
#endif
    return (*(array_t[t] + address(m, n)));
}


template <class T, class Derived>
inline T& array_base<T, Derived> :: operator()(const uint t, int m_in, int n_in)
{
    uint m = (m_in > 0 ? m_in : My + m_in) % My;
    uint n = (n_in > 0 ? n_in : Nx + n_in) % Nx;
#ifdef DEBUG
    check_bounds(m, n);
#endif
    return (*(array_t[t] + address(m, n)));
}


template <class T, class Derived>
inline T array_base<T, Derived> :: operator()(int m_in, int n_in) const
{
    uint m = (m_in > 0 ? m_in : My + m_in) % My;
    uint n = (n_in > 0 ? n_in : Nx + n_in) % Nx;
    //if(n_in < 0)
    //    cout << "n_in = " << n_in << "-> " << n << "\n";
#ifdef DEBUG
    check_bounds(m, n);
#endif
    return (*(array + address(m, n)));
}


template <class T, class Derived>
inline T& array_base<T, Derived> :: operator()(int m_in, int n_in)
{
    uint m = (m_in > 0 ? m_in : My + m_in) % My;
    uint n = (n_in > 0 ? n_in : Nx + n_in) % Nx;
    //if(n_in < 0)
    //    cout << "n_in = " << n_in << "-> " << n << "\n";
#ifdef DEBUG
    check_bounds(m, n);
#endif
    return (*(array + address(m, n)));
}

#endif // PERIODIC

/// @brief set tlev=0 to rhs
template <class T, class Derived>
Derived& array_base<T, Derived> :: operator=(const T& rhs)
{
    uint n{0}, m{0};
    for(m = 0; m < My; m++){
        for(n = 0; n < Nx; n++){
            (*this)(0, m, n) = rhs;
        }
    }
    return static_cast<Derived &>(*this);
}


/// @brief Copy memory pointed to by rhs.array_t[0] to array_t[0]
template <class T, class Derived>
Derived& array_base<T, Derived> :: operator=(const array_base<T, Derived>& rhs)
{
    if( (void*) this == (void*) &rhs)
        return static_cast<Derived &>(*this);
    check_bounds(rhs.get_my(), rhs.get_nx());
    memcpy(array, rhs.get_array(0), sizeof(T) * Nx * My);
    return static_cast<Derived &>(*this);
}


/// @brief Move result of temporary object to LHS
/// @detailed Swap array and array_t of this with rhs
template <class T, class Derived>
array_base<T, Derived>& array_base<T, Derived>::operator=(array_base<T, Derived>&& rhs)
{
    if( (void*) this == (void*) &rhs)
        return *this;
    check_bounds(rhs.get_tlevs(), rhs.get_my(), rhs.get_nx());

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
	if ( (void*) this == (void*) &rhs) 
        throw operator_err(string("array_base<T>& array_base<T>::operator*=(const array_base<T>& rhs): this == rhs\n"));

	check_bounds(rhs.get_tlevs(), rhs.get_my(), rhs.get_nx());
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
	if ( (void*) this == (void*) &rhs) 
        throw operator_err(string("array_base<T>& array_base<T>::operator-=(const array_base<T>& rhs): this == rhs\n"));
	check_bounds(rhs.get_tlevs(), rhs.get_my(), rhs.get_nx());

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
	if ( (void*) this == (void*) &rhs) 
        throw operator_err(string("array_base<T>& array_base<T>::operator*=(const array_base<T>& rhs): this == rhs\n"));

	check_bounds(rhs.get_tlevs(), rhs.get_my(), rhs.get_nx());
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
    check_bounds(t_src, 0, 0);
    check_bounds(t_dst, 0, 0);
    memcpy(array_t[t_dst], array_t[t_src], sizeof(T) * My * Nx);
}


/// @brief Copy memory from another array to this array
/// @param t_dst time level array_t[t_dst]
/// @param src Source for copy operation
/// @param t_src Time level to be copied from src
template <class T, class Derived>
void array_base<T, Derived>::copy(const uint t_dst, const Derived& src, const uint t_src)
{
    check_bounds(t_src, 0, 0);
    check_bounds(t_dst, 0, 0);
    memcpy(array_t[t_dst], src.array_t[t_src], sizeof(T) * My * Nx);
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
        memset(array_t[t_src], 0, My * Nx * sizeof(T));
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
    memset(array_t[0], 0, sizeof(T) * My * Nx);
}

#endif//ARRAY_BASE_H
// End of file array_base.h