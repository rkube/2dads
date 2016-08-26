/*
 * Time integration using the new boundary value array
 */


#include <iostream>
#include "slab_bc.h"
//#include "diagonstics.h"
#include "output.h"

using namespace std;

void print22(boost::property_tree::ptree const& pt)
{
    for(auto it : pt)
    {
        std::cout << it.first << ": " << it.second.get_value<std::string>() << std::endl;
        print22(it.second);
    }
}

int main(void)
{
    slab_config_js my_config(std::string("input.json"));

    std::cout << "main: " << my_config.get_runnr() << std::endl;

    std::cout << "runnr = " << my_config.get_runnr() << std::endl;
    std::cout << "xlef = "<< my_config.get_xleft() << std::endl;


    twodads::bvals_t<twodads::real_t> bv_theta(my_config.get_bvals_theta());

    //std::cout << "I am runnr " << my_config.get_runnr() << std::endl;
}
