/*
 * Convenient class to check array bounds
 */

#ifndef CHECK_BOUNDS_H
#define CHECK_BOUNDS_H

class check_bounds{
    public:
        //check_bounds() = delete;
        check_bounds(unsigned int nx, unsigned int my) : tlevs(0), Nx(nx), My(my) {};
        check_bounds(unsigned int t, unsigned int nx, unsigned int my) : tlevs(t), Nx(nx), My(my) {};
        inline bool operator()(unsigned int nx, unsigned int my) const
        {
            if (nx > Nx)
                return false;
            if (my > My)
                return false;
            return true;
        }
        inline bool operator()(unsigned int t, unsigned int nx, unsigned int my) const
        {
            if (t > tlevs)
                return false;
            if (nx > Nx)
                return false;
            if (my > My)
                return false;
            return true;
        }

    private:
        const unsigned int tlevs;
        const unsigned int Nx;
        const unsigned int My;
};

#endif //CHECK_BOUNDS_H
