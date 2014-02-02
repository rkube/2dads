#include <iostream>
//#include "include/cuda_array2.h"
#include "include/diag_array.h"


using namespace std;

int main(void)
{
    const unsigned int Nx = 8;
    const unsigned int My = 8;
    const unsigned int tlevs = 1;

    cuda_array<double> arr1(tlevs, Nx, My);
    arr1 = 4.0;

    cuda_array<double> arr2(tlevs, Nx, My);
    arr2 = 7.1;
    
    diag_array<double> da1(arr1);
    diag_array<double> da2(arr2);
    
    //for(int i = -int(da1.get_nx()); i < 0; i++)
    for(int i = int(da1.get_nx()); i < 2 * int(da1.get_nx()); i++)
    {
        da1(i,3) = -2.7;
        da2(3,i) = 5.7;
    }
    cout << "darr = " << da1 << "\n";
    cout << "darr = " << da2 << "\n";
   
    cout << "funny stuff: " << (da1 + da2).get_mean()<< "\n"; 

    cout << "Accessing da(-1, -2): " << da1(-1, -2) << "\n";


}
