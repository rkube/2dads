#include <iostream>
#include <cmath>
#include "cuda_darray.h"

using namespace std;

// Test the gpu diag array
//
// Routines:
//
// bar:   get poloidal mean
// tilde: get poloidal fluctuation around mean

int main(void)
{
    constexpr unsigned int Nx{16};
    constexpr unsigned int My{16};

    //cout << "Enter Nx: " << endl;
    //cin >> Nx;
    //cout << "Enter My: " << endl;
    //cin >> My;

    const double Lx = 1.0;
    const double Ly = 1.0;

    const double dx = 2. * Lx / ((double) Nx);
    const double dy = 2. * Ly / ((double) My);

    cout << "creating cuda_array<double>" << endl;
    cuda_array<double> carr(1, My, Nx);

    double x = 0.0;
    double y = 0.0;
    for(uint m = 0; m < carr.get_my(); m++)
    {
        y = -Ly + ((double) m) * dy;
        for(uint n = 0; n < carr.get_nx(); n++)
        {
            x = -Lx + ((double) n) * dx;
            carr(0, m, n) = 0.3 * x + sin(2.0 * M_PI * y);
        }
    }
    carr.copy_host_to_device();
    cout << "carr = \n" << carr << endl;
    cout << "creating cuda_darray<double>" << endl;
    cuda_darray<double> darr(carr);

    cout << "darr = \n" << darr << endl;
    double profile[Nx];

//    darr.get_profile(profile);
//
//    cout << "main(): profile = " << endl;
//    for(unsigned int n = 0; n < Nx; n++)
//    	cout << profile[n] << "\t";
//    cout << endl;

    cout << "max = " << darr.get_max() << endl;
    cout << "min = " << darr.get_min() << endl;
    cout << "sum = " << darr.get_sum() << endl;
    cout << "mean = "<< darr.get_mean() << endl;

    // This calls one full constructor and
    // move constructors of cuda_darray and cuda_array(delegated from cuda_darray)
    cuda_darray<double> darr_tilde(std::move(darr.tilde()));
    cout << darr_tilde;
    cout << "profile of tilde_array: ";
    darr_tilde.get_profile(profile);
    for(unsigned int n = 0; n< Nx; n++)
    	cout << profile[n] << "\t";
    cout << endl;

    // This calls one full constructor and
    // move constructors of cuda_darray and cuda_array(delegated from cuda_darray)
    cuda_darray<double> darr_bar(std::move(darr.bar()));
    cout << darr_bar;
    cout << "profile of bar_array: ";
    darr_bar.get_profile(profile);
    for(unsigned int n = 0; n< Nx; n++)
    	cout << profile[n] << "\t";
    cout << endl;

}
