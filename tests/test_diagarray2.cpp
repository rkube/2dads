#include <iostream>
#include <ctime>
#include <cstdio>
#include "include/diag_array.h"

using namespace std;


///
/// Test speedup for tilde, bar functions using threads
///

int main(void)
{
    const unsigned int Nx = 1024;
    const unsigned int My = 1024;
    const unsigned int nthreads = 2;

    double Lx = 2.0;
    double Ly = 2.0;
    double x = 0.0;
    double y = 0.0;
    double dx = Lx / double(Nx); 
    double dy = Lx / double(My);
    constexpr double PI{3.1415926};

    std::clock_t start;
    double duration;

    diag_array<double> da1(nthreads, 1, Nx, My);

    for(int n = 0; n < int(Nx); n++)
    {
        x = double(n) * dx;
        for(int m = 0; m < int(My); m++)
        {
            y = double(m) * dy;
            da1(n, m) = x + 0.38 * sin(2.0 * PI * y / Ly);
        }
    }

    cout << "Computing 1000 tilde with " << nthreads << " threads\n";
    start = std::clock();
    for(int i = 0; i < 1000; i++)
        da1.tilde_t();
    duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
    cout << "..." << duration << " seconds\n";

    cout << "Computing 1000 tilde with " << 1 << " threads\n";
    start = std::clock();
    for(int i = 0; i < 1000; i++)
        da1.tilde();
    duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
    cout << "..." << duration << " seconds\n";

    return(0);
    
}
