/*
 * Solve the diffusion eqations
 */

#include <iostream>
#include "slab_bc.h"

int main(void)
{
    size_t Nx{16};
    size_t My{16};
    cout << "Enter Nx: ";
    cin >> Nx;

    cout << "Enter My: ";
    cin >> My;

    cuda::bvals_t<double> my_bvals{cuda::bc_t::bc_dirichlet, cuda::bc_t::bc_dirichlet,
                                   cuda::bc_t::bc_periodic, cuda::bc_t::bc_periodic,
                                   0.0, 0.0, 0.0, 0.0};
    cuda::slab_layout_t my_geom(0.0, 1.0 / double(Nx), 0.0, 1.0 / double(My), Nx, 0, My, 2, cuda::grid_t::cell_centered);

    {
        slab_bc my_slab(my_geom, my_bvals);


    }

}
