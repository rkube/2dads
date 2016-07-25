/*
 * Test the new and awesome no ghost point datatype with boundary condition support
 */

#include <iostream>
#include <cuda_array_bc_nogp.h>

using namespace std;

int main(void)
{
    constexpr size_t Nx{128};
    constexpr size_t My{32};

    cuda::bvals<double> my_bvals{cuda::bc_t::bc_dirichlet, cuda::bc_t::bc_dirichlet, cuda::bc_t::bc_periodic, cuda::bc_t::bc_periodic,
                                 0.0, 0.0, 0.0, 0.0};
    cuda::slab_layout_t my_geom(0.0, 1.0 / cuda::real_t(Nx), 0.0, 1.0 / cuda::real_t(My), Nx, 0, My, 2);

    cuda_array_bc_nogp<double> my_ca(my_geom, my_bvals, size_t(1));
    my_ca.evaluate_device(0);
    //my_ca.enumerate();
    my_ca.copy_device_to_host();
    my_ca.dump_full();
}
