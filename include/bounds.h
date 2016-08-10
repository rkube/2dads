/*
 * Convenient class to check array bounds
 */

#ifndef CHECK_BOUNDS_H
#define CHECK_BOUNDS_H

#include "error.h"
#include <sstream>

/// Functor class that checks if arguments are within initialized bounds

class bounds{
    public:
        //check_bounds() = delete;
        bounds(size_t _my, size_t _nx) : tlevs(0), My(_my), Nx(_nx) {};
        bounds(size_t _t, size_t _my, size_t _nx) : tlevs(_t), My(_my), Nx(_nx) {};
        bool operator()(const size_t my, const size_t nx) const
        {
            if (my > My)
            {
                std::stringstream err_str;
                err_str << "bounds::operator()    ";
            	err_str << "Out of bounds:" << my << ">=" << My << "\n";
            	throw out_of_bounds_err(err_str.str());
            }
            if (nx > Nx)
            {
                std::stringstream err_str;
                err_str << "bounds::operator()    ";
            	err_str << "Out of bounds:" << nx << ">=" << Nx << "\n";
            	throw out_of_bounds_err(err_str.str());
            }
            return true;
        }
        bool operator()(const size_t t, const size_t my, const size_t nx) const
        {
        	operator()(my, nx);
            if (t > tlevs)
            {
                std::stringstream err_str;
                err_str << "bounds::operator()    ";
            	err_str << "Out of bounds:" << t << ">" << tlevs << "\n";
            	throw out_of_bounds_err(err_str.str());
            }
            return true;
        }

    private:
        const size_t tlevs;
        const size_t My;
        const size_t Nx;
};

#endif //CHECK_BOUNDS_H
