/*
 *
 * Test overloaded arithmetic operators of cuda_darray
 *
 * -> return types
 * -> correct use of move constructors
 * -> concatenation of arithmetic operators
 */

#include <iostream>
#include "cuda_darray.h"

using namespace std;


int main(void)
{
	constexpr unsigned int Nx{8};
	constexpr unsigned int My{8};

	cuda_darray<double> a1(My, Nx);
	cuda_darray<double> a2(My, Nx);

	a1 = 1.0;
	a2 = 2.0;
	cout << "a1:\n" << a1 << endl;
	cout << "a2:\n" << a2 << endl;

	cout << "assigning a1 *= a2" << endl;
	a1 *= a2;


	cout << "a1 = " << a1 << endl;
	cout << "a2 = " << a2 << endl;
	//cuda_darray<double> a3(a1+a2);
	cout << "===================================================" << endl;
	cout << "(a1 * a2).sum() = " << (a1 * a2).get_sum() << endl;
	cout << "(a1 * a2).mean() = " << (a1 * a2).get_mean() << endl;
	cout << "(a1 * a2).min() = " << (a1 * a2).get_min() << endl;
 	return(0);
}
