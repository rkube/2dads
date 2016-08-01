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

    cuda::slab_layout_t my_geom(0.0, 1.0 / cuda::real_t(Nx), 0.0, 1.0 / cuda::real_t(My), Nx, 0, My, 2);
    cuda::bvals_t<double> my_bvals{cuda::bc_t::bc_dirichlet, cuda::bc_t::bc_dirichlet, cuda::bc_t::bc_periodic, cuda::bc_t::bc_periodic,
                                   0.0, 0.0, 0.0, 0.0};
    cuda_array_bc_nogp<my_allocator_device<double> > my_ca(my_geom, my_bvals, size_t(1));
    cout << "exiting main()" << endl;
    my_ca.evaluate([=] __device__ (const size_t n, const size_t m, const cuda::slab_layout_t geom) -> cuda::real_t {return(3.2);}, 0);
    //my_ca.enumerate();
    my_ca.copy_device_to_host();
    my_ca.dump_full();
}
