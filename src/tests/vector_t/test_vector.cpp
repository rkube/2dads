/*
 * Test the vector routines
 */


#include <iostream>
#include "2dads_types.h"
#include "cuda_array_bc_nogp.h"
#include "utility.h"

using namespace std;
#ifdef HOST
using real_arr = cuda_array_bc_nogp<double, allocator_host>;
#endif

#ifdef DEVICE
using real_arr = cuda_array_bc_nogp<double, allocator_device>;
#endif

int main(void)
{
    twodads::slab_layout_t geom(-1.0, 0.25, -1.0, 0.25, 8, 0, 8, 2, twodads::grid_t::cell_centered);

    real_arr arr1(geom, twodads::bvals_t<double>(), 1);
    real_arr arr2(geom, twodads::bvals_t<double>(), 1);


    std :: cout << "arr1 = " << std::endl;
    arr1.apply([=] LAMBDACALLER(twodads::real_t input, const size_t n, const size_t m,twodads::slab_layout_t geom) -> twodads::real_t
        {
            return(1.0);
        }, 0);

    utility :: print(arr1, 0, std::cout);

    std :: cout << "arr2 = " << std::endl;
    arr2.elementwise([=] LAMBDACALLER (twodads::real_t lhs, twodads::real_t rhs) -> twodads::real_t 
        {return (lhs + 2.2 * rhs);}, arr1, 0, 0);

    arr2.set_transformed(0, true);
    utility :: print(arr2, 0, std::cout);
}