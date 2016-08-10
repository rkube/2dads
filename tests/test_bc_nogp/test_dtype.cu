/*
 * Test the new and awesome no ghost point datatype with boundary condition support
 */

#include <iostream>
#include <cuda_array_bc_nogp.h>

using namespace std;

int main(void)
{
    using value_t = double;

    size_t Nx{128};
    size_t My{32};
    const size_t tlevs{4};
    cout << "Enter Nx: " << endl;
    cin >> Nx;
    cout << "Enter My: " << endl;
    cin >> My;

    cuda::slab_layout_t my_geom(0.0, 1.0 / cuda::real_t(Nx), 0.0, 1.0 / cuda::real_t(My), Nx, 0, My, 2, tlevs, cuda::grid_t::cell_centered);
    cuda::bvals_t<value_t> my_bvals{cuda::bc_t::bc_dirichlet, cuda::bc_t::bc_dirichlet, cuda::bc_t::bc_periodic, cuda::bc_t::bc_periodic,
                                   0.0, 0.0, 0.0, 0.0};
    cout << "Entering scopy" << endl;
    {
        cuda_array_bc_nogp<value_t, allocator_device> vd (my_geom, my_bvals);
        vd.evaluate([=] __device__ (const size_t n, const size_t m, const cuda::slab_layout_t geom) -> value_t {return(3.2);}, 0);
        //vh.evaluate([=] (const size_t n, const size_t m, const cuda::slab_layout_t geom) -> value_t {return(3.2);}, 0);

        // Create a host copy and print the device data
        //cuda_array_bc_nogp<value_t, allocator_host> vh = utility :: create_host_vector(vd);
        //for(size_t t = 0; t < tlevs; t++)
        //{
        //    //vh.print(t);
        //} 
    }
    cout << "Leaving scope and exiting" << endl;
    cudaDeviceReset();
}
