/*
 * Test Fourier transformation in y direction
 *
 */

#include <iostream>
#include <cuda_array_bc.h>

using namespace std;

int main(void){
	cout << "Hello, World!" << endl;

    constexpr unsigned int Nx{16};
    constexpr unsigned int My{8};

    cuda::bvals<double> my_bvals{cuda::bc_t::bc_periodic, cuda::bc_t::bc_periodic, cuda::bc_t::bc_periodic, cuda::bc_t::bc_periodic, 0.0, 1.0, 0.0, 0.0};
    cuda::slab_layout_t my_geom(0.0, 1.0 / double(My), 0.0, 1.0 / double(Nx), 0.001, My, Nx);
	cuda_array_bc<double> my_ca(1, Nx, My, my_bvals, my_geom);

    my_ca.init_dft();
    my_ca.evaluate_device(0);
	my_ca.copy_device_to_host();
    my_ca.dump_full();
    cout << "===========================================================================================" << endl;

    my_ca.dft_r2c(0);
	my_ca.copy_device_to_host();
	my_ca.dump_full();

    cout << "===========================================================================================" << endl;
    my_ca.dft_c2r(0);
    my_ca.normalize(0);
    my_ca.copy_device_to_host();
    cout << my_ca << endl;
    my_ca.dump_full();

    return(0);
}
