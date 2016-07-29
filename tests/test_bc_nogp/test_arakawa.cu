/*
 * Test Arakawa bracket operators
 *
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include "slab_bc.h"

using namespace std;


int main(void){
    constexpr size_t Nx{1024};
    constexpr size_t My{1024};

    stringstream fname;

    cuda::slab_layout_t my_geom(-10.0, 20.0 / double(Nx), -10.0, 20.0 / double(My), Nx, 0, My, 2);
    cuda::bvals_t<double> my_bvals{cuda::bc_t::bc_dirichlet, cuda::bc_t::bc_dirichlet, cuda::bc_t::bc_periodic, cuda::bc_t::bc_periodic,
        0.0, 0.0, 0.0, 0.0};

    slab_bc my_slab(my_geom, my_bvals);

    //cerr << "Initializing fields..." << endl;
    my_slab.initialize_invlaplace(test_ns::field_t::arr1);
    // Print input to inv_laplace routine into array arr1_nx.dat
    fname << "test_laplace_arr1_" << Nx << ".dat";
    my_slab.print_field(test_ns::field_t::arr1, fname.str());

    my_slab.dft_r2c(test_ns::field_t::arr1, 0);

    my_slab.invert_laplace(test_ns::field_t::arr2, test_ns::field_t::arr1, size_t(0));
    my_slab.dft_c2r(test_ns::field_t::arr2, size_t(0));

    fname.str(string(""));
    fname << "test_laplace_arr2_" << Nx << ".dat";

    my_slab.print_field(test_ns::field_t::arr2, fname.str());
    cudaDeviceReset();
}

