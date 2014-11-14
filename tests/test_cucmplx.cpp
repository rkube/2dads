#include <iostream>
#include "cucmplx.h"

using namespace std;

int main(void)
{
    CuCmplx<double> z1(1.0, 1.0);
    CuCmplx<double> z2(0.1, 0.1);
    CuCmplx<double> z3(10.0, 10.0);
    
    double C = 3.0;
    cout << "1 + 3.0 * (10.0 + 0.1) = " << 1. + 3. * (10. + .1) << "\n";
    CuCmplx<double> result(0.0, 0.0);
    result += ((z2 + z3) * C) + z1;
    cout << "z1 + (z2 + z3) = " << ((z2 + z3) * C) + z1 << "\n";
    cout << z1 + ((z2 + z3) * C) << "\n";

    return(0);

}
