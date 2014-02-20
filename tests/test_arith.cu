/*
 * test_arith.cpp
 *
 * Created on: Dec 25, 2013
 *     Author: rku000
 *
 *
 * Test operator+=, operator-=, operatpr*= for the cuda_array
 * base class.
 *
 * When created with multiple time levels, operator[+-*]= still act only on
 * the tlev[0], i.e. the array stored in array_t_h[0].
 *
 * Plan:
 * create two arrays with 2 time levels.
 * Initialize arrays with constant values. Check if operators only operate on
 * data pointed to by array_t_h[0].
 * Advance time levels.
 * Verify that operators again only operate on data pointed to by array_t_h[0].
 */


#include <iostream>
#include "include/cuda_array3.h"

using namespace std;

int main(void)
{
    const int Nx = 8;
    const int My = 8;
    const int tlevs = 3;
    //cuda_array<CuCmplx<double>, double> arr1(tlevs, Nx, My);
    cuda_array<double, double> arr1(tlevs, Nx, My);

    arr1.set_t(4.0, 0);
    arr1.set_t(5.0, 1);
    arr1.set_t(6.0, 2);
    //arr2.set_all(2.0);
    cout << "arr1 = " << arr1 << "\n";

    arr1.advance();
    cout << "arr1 = " << arr1 << "\n";

    arr1.advance();
    cout << "arr1 = " << arr1 << "\n";


    /*
    arr1 = 1.0;
    arr2 = 3.0;
    cout << "arr1=\n" << arr1 << "\n";
    cout << "arr2=\n" << arr2 << "\n";

    arr1 -= arr2;
    cout << "arr1 -= arr2\narr1 = " << arr1 << "\n";

    arr2.advance();
    cout << "arr2.advance()\narr 2 = " << arr2 << "\n";

    arr1 -= arr2;
    cout << "arr1 -= arr2\narr1 = " << arr1 << "\n";
   
    arr1.advance();
    arr1 = 1.0;
    cout << "arr1=\n" << arr1 << "\n";
    arr1.advance();
    arr1 = 2.0;
    cout << "arr1=\n" << arr1 << "\n";
    */
    return(0);
}

