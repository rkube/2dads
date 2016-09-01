/*
 * Test the new and awesome no ghost point datatype with boundary condition support
 */

#include <iostream>
#include <cuda_array_bc_nogp.h>
#include <utility.h>

using namespace std;

#ifndef HOST
#ifndef DEVICE
#warning "Please specify -DDISPATCH=HOST or -DDISPATCH=DEVICE"
#endif
#endif


#ifdef HOST
#warning "HOST specified"
#endif

#ifdef DEVICE
#warning "DEVICE specified"
#endif

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

    twodads::slab_layout_t my_geom(0.0, 1.0 / twodads::real_t(Nx), 0.0, 1.0 / twodads::real_t(My), Nx, 0, My, 2, twodads::grid_t::cell_centered);
    twodads::bvals_t<value_t> my_bvals{twodads::bc_t::bc_dirichlet, twodads::bc_t::bc_dirichlet, 0.0, 0.0}; 
    cout << "Entering scope" << endl;
    {
        cuda_array_bc_nogp<value_t, allocator_host> vh (my_geom, my_bvals, tlevs);
        vh.apply([] (value_t dummy, const size_t n, const size_t m, const twodads::slab_layout_t geom) -> value_t {return(1.2);}, 0);
        vh.apply([] (value_t dummy, const size_t n, const size_t m, const twodads::slab_layout_t geom) -> value_t {return(2.2);}, 1);
        vh.apply([] (value_t dummy, const size_t n, const size_t m, const twodads::slab_layout_t geom) -> value_t {return(3.2);}, 2);
        vh.apply([] (value_t dummy, const size_t n, const size_t m, const twodads::slab_layout_t geom) -> value_t {return(4.2);}, 3);

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

    }
    cout << "Leaving scope and exiting" << endl;
}
