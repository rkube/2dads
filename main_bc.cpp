/*
 * Time integration using the new boundary value array
 */


#include <iostream>
#include "slab_bc.h"
//#include "diagonstics.h"
#include "output.h"

using namespace std;


int main(void)
{
    slab_config_js my_config(std::string("input.json"));
    {
        slab_bc my_slab(my_config);

        std::cout << "I am runnr " << my_config.get_runnr() << std::endl;

        my_slab.initialize();
    }
}
