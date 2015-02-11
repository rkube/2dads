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
        bounds(unsigned int my, unsigned int nx) : tlevs(0), My(my), Nx(nx) {};
        bounds(unsigned int t, unsigned int my, unsigned int nx) : tlevs(t), My(my), Nx(nx) {};
        bool operator()(const unsigned int my, const unsigned int nx) const
        {
            if (my > My)
            {
            	stringstream err_str;
            	err_str << "Out of bounds:" << my << ">=" << My << "\n";
            	throw out_of_bounds_err(err_str.str());
            }
            if (nx > Nx)
            {
            	stringstream err_str;
            	err_str << "Out of bounds:" << nx << ">=" << Nx << "\n";
            	throw out_of_bounds_err(err_str.str());
            }
            return true;
        }
        bool operator()(const unsigned int t, const unsigned int my, const unsigned int nx) const
        {
        	operator()(my, nx);
            if (t > tlevs)
            {
            	stringstream err_str;
            	err_str << "Out of bounds:" << t << ">" << t << "\n";
            	throw out_of_bounds_err(err_str.str());
            }
            return true;
        }

    private:
        const unsigned int tlevs;
        const unsigned int My;
        const unsigned int Nx;
};

#endif //CHECK_BOUNDS_H
