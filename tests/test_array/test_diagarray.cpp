#include <iostream>
#include "diag_array.h"


using namespace std;

int main(void)
{
    const unsigned int Nx = 16;
    const unsigned int My = 16;
    const unsigned int tlevs = 1;
    const unsigned int nthreads = 8;

    // Create cuda arrays
    cuda_array<double, double> arr1(tlevs, Nx, My);
    arr1 = 4.0;

    cuda_array<double, double> arr2(tlevs, Nx, My);
    arr2 = 7.1;
   
    // Initialize diagnostic arrays from cuda arrays 
    diag_array<double> da1(arr1);
    diag_array<double> da2(arr2);

    // Initialize from another diag_Array
    diag_array<double> da3(move(da1.bar()));

    cout << "da1 = " << da1 << "\n";
    cout << "da3 = " << da3 << "\n";


    da1.set_numthreads(nthreads);
    da2.set_numthreads(nthreads);

    // Test index wrapping 
    for(int n = int(da1.get_nx()); n < 2 * int(da1.get_nx()); n++)
    {
        da1(n,3) = -2.7;
        da2(3,n) = 5.7;
    }
    // And access to negative indices
    cout << "Accessing da(-1, -2): " << da1(-1, -2) << "\n";
  
    // Test for correct return values of operator[+-*] member functions
    // of array_base
    cout << "funny stuff: " << (da1 + da2).get_mean()<< "\n"; 

    // Test derivation functions
    // First derivative: initialize linear x profile, derivative should be constant
    // Execpt for boundaries due to periodic boundary conditions
   
    double Lx = 2.0;
    double Ly = 2.0;
    double x = 0.0;
    double y = 0.0;
    double dx = Lx / double(Nx); 
    double dy = Lx / double(My);
    constexpr double PI{3.1415926};
    for(int n = 0; n < int(Nx); n++)
    {
        x = double(n) * dx;
        for(int m = 0; m < int(My); m++)
            da1(n, m) = x*x*x;
    }
    //cout << "x^3 profile: " << da1 << "\n";
    diag_array<double> res(da1.d3_dx3(Lx));
    cout << "res = " << res << "\n";

    //for(int n = 0; n < int(Nx); n++)
    //    for(int m = 0; m < int(My); m++)
    //    {
    //        y = double(m) * dy;
    //        da1(n, m) = y * y * y;
    //    }
    //cout << "y^3 profiles: " << da1 << "\n";
    //diag_array<double> resy(da1.d3_dy3(Lx));
    //cout << "res = " << resy << "\n";

    // Test profile function: f(x,y) = x + 0.38 * sin(2 pi y / Ly)
    cout << "Comparing mean, max, min : single vs multithread on sinusoidal profile:\n";
    for(int n = 0; n < int(Nx); n++)
    {
        x = double(n) * dx;
        for(int m = 0; m < int(My); m++)
        {
            y = double(m) * dy;
            da1(n, m) = x + 0.38 * sin(2.0 * PI * y / Ly);
        }
    }

    cout << "da1 = " << da1 << "\n";
    cout << "da1.mean() = " << da1.get_mean() << "\tda1.mean_t() = " << da1.get_mean_t() << "\n";
    cout << "da1.max() = " << da1.get_max() << "\tda1.max_t() = " << da1.get_max_t() << "\n";
    cout << "da1.min() = " << da1.get_min() << "\tda1.min_t() = " << da1.get_min_t() << "\n";


    cout << "Computing profile\n";
    cout << "Singlethreaded:\n";
    da1.dump_profile();

    cout << "\nMultithreaded\n";
    da1.dump_profile_t();

    cout << "Mean value array\n";
    cout << "Single threaded:\n" << da1.bar() << "\n";
    cout << "Multi threaded:\n" << da1.bar_t() << "\n";

    cout << "Fluctuations\n";
    cout << "Single threaded:\n" << da1.tilde() << "\n";
    cout << "Multi threaded:\n" << da1.tilde_t() << "\n";
}
