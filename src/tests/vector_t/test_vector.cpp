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
    twodads::slab_layout_t geom(-1.0, 0.25, -1.0, 0.25, 8, 0, 8, 0, twodads::grid_t::cell_centered);
    twodads::bvals_t<double> bvals;

    real_arr arr(geom, bvals, 1);

    utility :: print(arr, 0, std::cout);
}