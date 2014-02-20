/*
 * Test spatial derivatives
 */


#include <iostream>
#include <include/slab_cuda.h>

using namespace std;

int main(void)
{
    slab_config my_config;
    my_config.consistency();

    slab_cuda slab(my_config);
    slab.initialize();

    cout << "Derivative in x  direction\n";
    slab.d_dx(twodads::field_k_t::f_theta_hat, twodads::field_k_t::f_theta_x_hat, 3);
    slab.dump_field(twodads::field_k_t::f_theta_x_hat);

    cout << "Derivative in y-direction\n";
    slab.d_dy(twodads::field_k_t::f_theta_hat, twodads::field_k_t::f_theta_y_hat, 3);
    slab.dump_field(twodads::field_k_t::f_theta_y_hat);


}
