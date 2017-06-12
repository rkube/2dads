/*
 * Convenient class to check array bounds
 */

#ifndef CHECK_BOUNDS_H
#define CHECK_BOUNDS_H

#include "error.h"
#include <sstream>

/// Class that checks if arguments are within initialized bounds

class bounds{
    /** 
     .. cpp:namespace-push:: bounds

     */

     /**
      .. cpp:class bounds

      Functor that implements checking for out-of-bounds errors.

    */
    public:

        /**
         .. cpp:function:: bounds(const size_t _my, const size_t _nx)

          :param const size_t _my: Max elements in y-direction. 
          :param const size_t _nx: Max elements in x-direction.

          Initializes bound functor with _nx and _my. Numbers larger than
          _nx and _my will be seen as out-of-bounds. Sets 1 as time levels.

        */
        bounds(const size_t _my, const size_t _nx) : tlevs(0), My(_my), Nx(_nx) {};

        /**
         .. cpp:function:: bounds(const size_t _my, const size_t _nx)

          :param const size_t _t: Max number of time levels.
          :param const size_t _my: Max elements in y-direction. 
          :param const size_t _nx: Max elements in x-direction.
          
          Initializes bound functor with _nx and _my. Numbers larger than
          _t, _nx and _my will be seen as out-of-bounds.

        */
        bounds(const size_t _t, const size_t _my, const size_t _nx) : tlevs(_t), My(_my), Nx(_nx) {};


        /**
         .. cpp:function:: operator() (const size_t my, const size_t nx) const

         :param const size_t my: Number of y discretization points to be compared against.
         :param const size_t nx: Number of y discretization points to be compared against.

         Throws an out_of_bounds_err when one of the passed arguments is larger
         than the value of the corresponding member.

        */
        bool operator()(const size_t my, const size_t nx) const
        {
            if (my > get_my())
            {
                std::stringstream err_str;
                err_str << "bounds::operator()    ";
            	err_str << "Out of bounds:" << my << ">=" << get_my() << "\n";
            	throw out_of_bounds_err(err_str.str());
            }
            if (nx > get_nx())
            {
                std::stringstream err_str;
                err_str << "bounds::operator()    ";
            	err_str << "Out of bounds:" << nx << ">=" << get_nx() << "\n";
            	throw out_of_bounds_err(err_str.str());
            }
            return true;
        }

        /**
         .. cpp:function:: operator() (const size_t t, const size_t my, const size_t nx) const

         :param const size_t t: Number of time levels to be compared against.
         :param const size_t my: Number of y discretization points to be compared against.
         :param const size_t nx: Number of y discretization points to be compared against.

         Throws an out_of_bounds_err when one of the passed arguments is larger
         than the value of the corresponding member.

        */
        bool operator()(const size_t t, const size_t my, const size_t nx) const
        {
        	operator()(my, nx);
            if (t > get_tlevs())
            {
                std::stringstream err_str;
                err_str << "bounds::operator()    ";
            	err_str << "Out of bounds:" << t << ">" << tlevs << "\n";
            	throw out_of_bounds_err(err_str.str());
            }
            return true;
        }

        /**
         .. cpp:function:: get_my() const

         Return my.

        */
        inline size_t get_my() const {return(My);};

        /**
         .. cpp:function:: get_nx() const

         Returns nx.

        */
        inline size_t get_nx() const {return(Nx);};

        /**
         .. cpp:function:: get_tlevs() const

         Returns tlevs.

        */
        inline size_t get_tlevs() const {return(tlevs);};

    private:
        const size_t tlevs;
        const size_t My;
        const size_t Nx;

    /**
     .. cpp:namespace-pop

    */
};

#endif //CHECK_BOUNDS_H
