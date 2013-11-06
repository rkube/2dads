/*
 *
 * test_slab.cpp
 *
 */

#include <iostream>
#include "include/slab_cuda.h"

using namespace std;

int main(void)
{
    slab_config my_config;
    cout << "Hello, World!\n";
    cout << "config.nx = " << my_config.get_nx() << ", My = " << my_config.get_my() << "\n";

    slab_cuda slab(my_config);
    return(0);
}
