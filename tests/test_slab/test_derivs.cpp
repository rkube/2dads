/*
 * Test spatial derivatives
 */


#include <iostream>
#include <slab_cuda.h>

using namespace std;

int main(void)
{
    slab_config my_config;
    my_config.consistency();

    slab_cuda slab(my_config);
    slab.initialize();

    slab.print_field(twodads::field_k_t::f_theta_hat);

    //cout << "Derivative in x-direction\n";
    //slab.d_dx_dy(twodads::field_k_t::f_theta_hat, twodads::field_k_t::f_theta_x_hat, twodads::field_k_t::f_theta_y_hat, 3);
    //slab.print_field(twodads::field_k_t::f_theta_x_hat);
    //slab.print_field(twodads::field_k_t::f_theta_y_hat);

//    slab.dft_c2r(twodads::field_k_t::f_theta_x_hat, twodads::field_t::f_theta_x, 0);
//    slab.dft_c2r(twodads::field_k_t::f_theta_y_hat, twodads::field_t::f_theta_y, 0);
//    cout << "\n\n\n\n";
//    cout << "theta:\n";
//    slab.print_field(twodads::field_t::f_theta);
//    cout << "theta_x: \n";
//    slab.print_field(twodads::field_t::f_theta_x);
//    cout << "theta_y: \n";
//    slab.print_field(twodads::field_t::f_theta_y);

}
