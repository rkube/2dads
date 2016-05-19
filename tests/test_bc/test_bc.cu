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

    cuda::bvals<double> my_bvals{cuda::bc_t::bc_dirichlet, cuda::bc_t::bc_dirichlet, cuda::bc_t::bc_periodic, cuda::bc_t::bc_periodic, 0.0, 1.0, 0.0, 0.0};
    slab_layout_t my_geom(0.0, 1.0 / 16., 0.0, 1.0 / 16., 0.001, 16, 16);
	cuda_array_bc<double> my_ca(2, 16, 16, my_bvals, my_geom);

	//my_ca.enumerate();
    my_ca.evaluate_device(0);
    my_ca.update_ghost_points(0);
	my_ca.copy_device_to_host();
	my_ca.dump_full();
	//cout << endl << endl;
	//cout << my_ca;

}
