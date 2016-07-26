/*
 * Attempt to solve the laplace equation with boundary conditions in x
 * using the cusparse tridiagonal solver with the new datatype
 *
 * Invert
 * g(x,y) = exp(-(x^2 + y^2) / 2)
 * \nabla^2 g(x,y) = f(x,y) 
 * where
 * f(x,y) = exp(-(x^2 + y^2) / 2) (-1 + x^2) (-1 + y^2) 
 *
 * Goal: Given f(x,y) find g(x,y)
 */


#include <iostream>
#include <cassert>
#include "slab_bc.h"
#include "derivatives.h"

using namespace std;

int main(void){
    constexpr size_t Nx{256};
    constexpr size_t My{256};
    cout << "Hello, world!" << endl; 

    cuda::slab_layout_t my_geom(-10.0, 20.0 / double(Nx), -10.0, 20.0 / double(My), Nx, 0, My, 2);
    cuda::bvals_t<double> my_bvals{cuda::bc_t::bc_dirichlet, cuda::bc_t::bc_dirichlet, cuda::bc_t::bc_periodic, cuda::bc_t::bc_periodic,
                                   0.0, 0.0, 0.0, 0.0};

    slab_bc my_slab(my_geom, my_bvals);

    //cerr << "Initializing fields..." << endl;
    my_slab.initialize_invlaplace(test_ns::field_t::arr1);
    my_slab.print_field(test_ns::field_t::arr1, string("arr1.dat")); 

    my_slab.dft_r2c(test_ns::field_t::arr1, 0);
    my_slab.print_field(test_ns::field_t::arr1, string("arr1_dft.dat")); 

    //arr1.init_inv_laplace();
    //arr1.copy_device_to_host();
    //cout << arr1 << endl;
    my_slab.invert_laplace(test_ns::field_t::arr2, test_ns::field_t::arr1, size_t(0));
    //my_slab.dft_c2r(test_ns::field_t::arr2, 0)
    my_slab.print_field(test_ns::field_t::arr2, string("arr2_dft.dat"));
}
