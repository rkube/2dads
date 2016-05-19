/*
 * main.cpp
 *
 *  Created on: May 13, 2016
 *      Author: ralph
 */


#include <iostream>
#include <cuda_array_bc.h>


using namespace std;

int main(void){
	cout << "Hello, World!" << endl;
    constexpr unsigned int Nx{64};
    constexpr unsigned int My{16};

    cuda::bvals<double> my_bvals{cuda::bc_t::bc_periodic, cuda::bc_t::bc_periodic, cuda::bc_t::bc_periodic, cuda::bc_t::bc_periodic, 0.0, 1.0, 0.0, 0.0};
    cuda::slab_layout_t my_geom(0.0, 1.0 / cuda::real_t(My), 0.0, 1.0 / cuda::real_t(Nx), 0.001, My, Nx);
	cuda_array_bc<double> my_ca(1, Nx, My, my_bvals, my_geom);

    my_ca.evaluate_device(0);
    //my_ca.update_ghost_points(0);
	my_ca.copy_device_to_host();
	my_ca.dump_full();
	//cout << endl << endl;
	//cout << my_ca;

}
