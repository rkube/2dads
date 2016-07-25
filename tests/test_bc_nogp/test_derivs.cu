/*
 * Test derivatives
 */

#include <iostream>
#include "slab_bc.h"

using namespace std;

int main(void)
{
    constexpr size_t Nx{16};
    constexpr size_t My{16};

    cuda::bvals_t<double> my_bvals{cuda::bc_t::bc_dirichlet, cuda::bc_t::bc_dirichlet, cuda::bc_t::bc_periodic, cuda::bc_t::bc_periodic, 0.0, 0.0, 0.0, 0.0};
    cuda::slab_layout_t my_geom(0.0, 1.0 / double(Nx), 0.0, 1.0 / double(My), Nx, 0, My, 2);
    //cuda::bvals_t<double> my_bvals{cuda::bc_t::bc_periodic, cuda::bc_t::bc_periodic, cuda::bc_t::bc_periodic, cuda::bc_t::bc_periodic, 0.0, 1.0, 0.0, 0.0};
    //cuda::slab_layout_t my_geom(0.0, 1.0 / double(Nx), 0.0, 1.0 / double(My), Nx, 0, My, 2);

    slab_bc my_slab(my_geom, my_bvals);

    my_slab.dump_arr1();

    my_slab.d_dx_dy(0);
    cout << "===========================================================================================" << endl;
    my_slab.dump_arr1x();
    //cout << "===========================================================================================" << endl;
    //my_slab.dump_arr1y();
}
