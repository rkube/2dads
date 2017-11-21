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
    twodads::bvals_t<double> bvals;

    constexpr twodads::real_t coeffs[4] = {0.5, 1.0, 10.0, 100.0};

    real_arr arr(geom, bvals, 4);

    // Initialize arr[3] = 3, arr[2] = 2, arr[1] = 1
    arr.apply([] LAMBDACALLER(twodads::real_t dummy, const size_t n, const size_t m, twodads::slab_layout_t geom) -> twodads::real_t
              { return(3.0); }, 3);
    arr.apply([] LAMBDACALLER(twodads::real_t dummy, const size_t n, const size_t m, twodads::slab_layout_t geom) -> twodads::real_t
              { return(2.0); }, 2);
    arr.apply([] LAMBDACALLER(twodads::real_t dummy, const size_t n, const size_t m, twodads::slab_layout_t geom) -> twodads::real_t
              { return(1.0); }, 1);

    // Accumulate arr[0] <- arr[3] + arr[2] + arr[1]
    arr.elementwise([=] LAMBDACALLER (twodads::real_t lhs, twodads::real_t rhs) -> twodads::real_t
                    {return(lhs + coeffs[3] * rhs);}, 0, 3);
    arr.elementwise([=] LAMBDACALLER (twodads::real_t lhs, twodads::real_t rhs) -> twodads::real_t
                    {return(lhs + coeffs[2] * rhs);}, 0, 2);
    arr.elementwise([=] LAMBDACALLER (twodads::real_t lhs, twodads::real_t rhs) -> twodads::real_t
                    {return(lhs + coeffs[1] * rhs);}, 0, 1);

    arr.elementwise([=] LAMBDACALLER (twodads::real_t lhs, twodads::real_t dummy) -> twodads::real_t
                    {return(lhs / coeffs[0]);}, 0, 0);
    
    std::cout << arr.is_transformed(0) << std::endl;
    utility :: print(arr, 0, std::cout);
}
