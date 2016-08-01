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
    constexpr size_t Nx{512};
    constexpr size_t My{512};

    stringstream fname;

    cuda::slab_layout_t my_geom(-1.0, 2.0 / double(Nx), -0.0, 2.0 / double(My), Nx, 0, My, 2);
    cuda::bvals_t<double> my_bvals{cuda::bc_t::bc_dirichlet, cuda::bc_t::bc_dirichlet, cuda::bc_t::bc_periodic, cuda::bc_t::bc_periodic,
        0.0, 0.0, 0.0, 0.0};

    {
        slab_bc my_slab(my_geom, my_bvals);

        //cerr << "Initializing fields..." << endl;
        my_slab.initialize_arakawa(test_ns::field_t::arr1, test_ns::field_t::arr2);
        // Print input to inv_laplace routine into array arr1_nx.dat
        fname << "test_arakawa_arr1_" << Nx << "_in.dat";
        my_slab.print_field(test_ns::field_t::arr1, fname.str());

        fname.str(string(""));
        fname << "test_arakawa_arr2_" << Nx << "_in.dat";
        my_slab.print_field(test_ns::field_t::arr2, fname.str());

        my_slab.arakawa(test_ns::field_t::arr1, test_ns::field_t::arr2, test_ns::field_t::arr3, size_t(0), size_t(0));

        fname.str(string(""));
        fname << "test_arakawa_arr3_" << Nx << "_out.dat";
        my_slab.print_field(test_ns::field_t::arr3, fname.str());
    }
    cudaDeviceReset();
}

