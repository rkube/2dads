/*
 * Test derivation kernels
 */

#include <iostream>
#include "slab_cuda.h"

using namespace std;

int main(void)
{
    slab_config my_config;
    my_config.consistency();

    slab_cuda slab(my_config);
    slab.initialize();

    slab.print_grids();
    //cout << "Enumerating:" << endl;
    //cout << "theta_x_hat" << endl;
    //slab.enumerate(twodads::field_k_t::f_theta_x_hat);
    //slab.print_field(twodads::field_k_t::f_theta_x_hat);

    //cout << "theta_x" << endl;
    //slab.enumerate(twodads::field_t::f_theta_x);
    //slab.print_field(twodads::field_t::f_theta_x);

    //cout << "Enumerating kernels for d/dx\n";
    //slab.d_dx_enumerate(twodads::field_k_t::f_theta_x_hat, 0);
    //slab.print_field(twodads::field_k_t::f_theta_x_hat);

    cout << "Enumerating kernels for d/dy\n";
    slab.d_dy_enumerate(twodads::field_k_t::f_theta_y_hat, 0);
    slab.print_field(twodads::field_k_t::f_theta_y_hat);

    //cout << "Enumerating kernels for inv_laplace\n";
    //slab.inv_laplace_enumerate(twodads::field_k_t::f_theta_hat, 3);
    //slab.print_field(twodads::field_k_t::f_tmp_hat);

    //cout << "Enumerating kernels for stiffk\n";
    //slab.integrate_stiff_enumerate(twodads::field_k_t::f_theta_hat, 2);
    //slab.print_field(twodads::field_k_t::f_theta_rhs_hat);
}
