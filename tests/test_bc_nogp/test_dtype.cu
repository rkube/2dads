/*
 * Test the new and awesome no ghost point datatype with boundary condition support
 */

#include <iostream>
#include <cuda_array_bc_nogp.h>
#include <utility.h>


#ifdef __CUDACC__
#warning main: compiling for device
#endif

#ifndef __CUDACC__
#warning main: compiling for host
#endif

using arr_real = cuda_array_bc_nogp<double, allocator_device>;

int main(void)
{
    using value_t = double;

    size_t Nx{16};
    size_t My{16};
    
    const size_t tlevs{2};
    //std::cout << "Enter Nx: " << std::endl;
    //std::cin >> Nx;
    //std::cout << "Enter My: " << std::endl;
    //std::cin >> My;

    twodads::slab_layout_t my_geom(0.0, 1.0 / twodads::real_t(Nx), 0.0, 1.0 / twodads::real_t(My), Nx, 0, My, 2, twodads::grid_t::cell_centered);
    twodads::bvals_t<value_t> my_bvals{twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_dirichlet, 0.0, 0.0};


    std::cout << "Entering scope" << std::endl;
    {
        arr_real vd (my_geom, my_bvals, tlevs);
        /*
        vd.apply([] LAMBDACALLER (value_t dummy, const size_t n, const size_t m, const twodads::slab_layout_t geom) -> value_t {return(1.2);}, 0);
        vd.apply([] LAMBDACALLER (value_t dummy, const size_t n, const size_t m, const twodads::slab_layout_t geom) -> value_t {return(2.2);}, 1);
        vd.apply([] LAMBDACALLER (value_t dummy, const size_t n, const size_t m, const twodads::slab_layout_t geom) -> value_t {return(3.2);}, 2);
        vd.apply([] LAMBDACALLER (value_t dummy, const size_t n, const size_t m, const twodads::slab_layout_t geom) -> value_t {return(4.2);}, 3);
        cuda_array_bc_nogp<value_t, allocator_host> vh = utility :: create_host_vector(vd);

        // Create a host copy and print the device data
        for(size_t t = 0; t < tlevs; t++)
        {
            cout << t << endl;
            utility :: print(vh, t, std::cout);
        } 
        //cout << "=================================== advance ========================" << endl;

        //vh.advance();
        //utility :: update_host_vector(vh, vd);
        //for(size_t t = 0; t < tlevs; t++)
        //{
        //    cout << t << endl;
        //    utility :: print(vh, t, std::cout);
        //} 
        */
    }
    std::cout << "Leaving scope and exiting" << std::endl;
    cudaDeviceReset();
}
