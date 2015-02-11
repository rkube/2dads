/*
 * Test spatial derivatives, to be called from python
 *
 *
 */

#include <iostream>
#include <slab_cuda.h>

using namespace std;

// Initialize a slab according to input.ini in the working directory
// of the script. write results, as computed by slab_cuda data type
// into the arrays out_theta, out_theta_x, out_theta_y
//
// p_theta = double[Nx * My]
// p_theta_x = double[Nx * My]
// p_theta_y = double[Nx * My]
//
// where the pointer to each array is taken from
//
// theta = np.ndarray([Nx * My], dtype='float64').ctypes.data 

extern "C" void test_derivs(void* p_theta, void* p_theta_x, void* p_theta_y)
{
    slab_config my_config;
    my_config.consistency();

    slab_cuda slab(my_config);
    slab.initialize();

    slab.get_data(twodads::field_t::f_theta, (double*) p_theta);
    slab.get_data(twodads::field_t::f_theta_x, (double*) p_theta_x);
    slab.get_data(twodads::field_t::f_theta_y, (double*) p_theta_y);

//    cout << "theta_hat=" << endl;
//    slab.print_field(twodads::field_k_t::f_theta_hat);
//    cout << "theta_x_hat=" << endl;
//    slab.print_field(twodads::field_k_t::f_theta_x_hat);
//    cout << "theta_y_hat=" << endl;
//    slab.print_field(twodads::field_k_t::f_theta_y_hat);

} 
