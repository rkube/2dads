/*
 * Convenient class to check array bounds
 */

#ifndef CHECK_BOUNDS_H
#define CHECK_BOUNDS_H

class check_bounds{
    public:
        //check_bounds() = delete;
        check_bounds(unsigned int my, unsigned int nx) : tlevs(0), My(my), Nx(nx) {};
        check_bounds(unsigned int t, unsigned int my, unsigned int nx) : tlevs(t), My(my), Nx(nx) {};
        inline bool operator()(unsigned int my, unsigned int nx) const
        {
            if (my > My)
                return false;
            if (nx > Nx)
                return false;
            return true;
        }
        inline bool operator()(unsigned int t, unsigned int my, unsigned int nx) const
        {
            if (t > tlevs)
                return false;
            if (my > My)
                return false;
            if (nx > Nx)
                return false;
            return true;
        }

    private:
        const unsigned int tlevs;
        const unsigned int My;
        const unsigned int Nx;
};

#endif //CHECK_BOUNDS_H
