#include <iostream>
#include "include/cuda_array2.h"
#include "include/diag_array.h"


using namespace std;

int main(void)
{
    const unsigned int Nx = 8;
    const unsigned int My = 8;
    const unsigned int tlevs = 1;

    cuda_array<double> arr1(tlevs, Nx, My);
    arr1 = 4.0;
    cout << "arr1 = \n" << arr1 << "\n";

    diag_array<double> darr(arr1);
    for(unsigned int i = 0; i < darr.get_nx(); i++)
        darr(3,i) = -2.7;
    for(unsigned int i = 0; i < darr.get_nx(); i++)
        cout << "profile(" << i << ") = " << darr.get_profile(i) << "\n";
    cout << "darr = " << darr << "\n";
    cout << "max(darr) = " << darr.get_max() << "\n";
    cout << "darr.max =" << darr.get_max() << "\n";
    cout << "darr.min =" << darr.get_min() << "\n";
    cout << "darr.mean = " << darr.get_mean() << "\n";

}
