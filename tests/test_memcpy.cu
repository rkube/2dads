/*
 *
 * test_output.cpp
 *
 * Test some memcpy methods for passing data to cuda
 *
 */

#include <iostream>
#include <stdio.h>
#include "vector_types.h"
#include "cuda_types.h"
#include "cufft.h"

using namespace std;


__global__
void d_dump_c(cuda::cmplx_t* alpha, uint N, uint width)
{
    printf("Hello from the GPU \n");
    
    for(int i = 0; i < N; i++){
        for(int j = 0; j < width; j++){
            //printf("alpha[%d][%d] = (%f, %f)\t", i, j, alpha[i][j].x, alpha[i][j].y);
            printf("alpha[%d][%d] = (%f, %f)\t", i, j, (alpha[i * width + j]).re(), (alpha[i * width + j]).im());
        }
        printf("\n");
    }
    
}

__global__
void d_dump_r(cuda::real_t* alpha, uint N, uint width)
{
    printf("Hello from the GPU \n");
    
    for(int i = 0; i < N; i++){
        for(int j = 0; j < width; j++){
            //printf("alpha[%d][%d] = (%f, %f)\t", i, j, alpha[i][j].x, alpha[i][j].y);
            printf("alpha[%d][%d] = (%f)\t", i, j, alpha[i * width + j]);
        }
        printf("\n");
    }
    
}


int main(void)
{
    //cuda::cmplx_t alpha_c[3][4] = {{make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(0.0, 0.0)},
    //                               {make_cuDoubleComplex(1.5, 0.0), make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(-0.5, 0.0), make_cuDoubleComplex(0.0, 0.0)},
    //                               {make_cuDoubleComplex(11.0 / 6.0, 0.0), make_cuDoubleComplex(3.0, 0.0), make_cuDoubleComplex(-1.5, 0.0), make_cuDoubleComplex(1.0/3.0, 0.0)}};
    cuda::cmplx_t alpha_c[3][4] = {{cuda::cmplx_t(1.0, 0.0), cuda::cmplx_t(1.0, 0.0), cuda::cmplx_t(0.0, 0.0), cuda::cmplx_t(0.0, 0.0)},
                                   {cuda::cmplx_t(1.5, 0.0), cuda::cmplx_t(1.0, 0.0), cuda::cmplx_t(-0.5, 0.0), cuda::cmplx_t(0.0, 0.0)},
                                   {cuda::cmplx_t(11.0 / 6.0, 0.0), cuda::cmplx_t(3.0, 0.0), cuda::cmplx_t(-1.5, 0.0), cuda::cmplx_t(1.0/3.0, 0.0)}};
    cuda::real_t alpha_r[3][4] = {{1.0, 1.0, 0.0, 0.0}, {1.5, 1.0, -0.5, 0.0}, {11.0/6.0, 3.0, -1.5, 1.0/3.0}};
    /*
    cuda::cmplx_t beta[3][3] = {{make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(0.0, 0.0)},
                                {make_cuDoubleComplex(2.0, 0.0), make_cuDoubleComplex(-1.0, 0.0), make_cuDoubleComplex(0.0, 0.0)},
                                {make_cuDoubleComplex(3.0, 0.0), make_cuDoubleComplex(-3.0, 0.0), make_cuDoubleComplex(1.0, 0.0)}};
    */
    cuda::cmplx_t* d_alpha_c;
    cuda::real_t* d_alpha_r;
    //cuda::cmplx_t** d_beta;

    cout << "sizeof(alpha_r) = " << sizeof(alpha_r) << "\n";
    cout << "sizeof(alpha_c) = " << sizeof(alpha_c) << "\n";
    cudaMalloc((void**) &d_alpha_r, sizeof(alpha_r));
    cudaMemcpy(d_alpha_r, &alpha_r, sizeof(alpha_r), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &d_alpha_c, sizeof(alpha_c));
    cudaMemcpy(d_alpha_c, &alpha_c, sizeof(alpha_c), cudaMemcpyHostToDevice);


    d_dump_r<<<1, 1>>>(d_alpha_r, 3, 4);
    d_dump_c<<<1, 1>>>(d_alpha_c, 3, 4);
    //cudaMemcpy(&sp, d_stiff_params, sizeof(cuda::stiff_params_t), cudaMemcpyDeviceToHost);
    //cout << "main: delta_t = " << sp.delta_t << "\n";
    cudaDeviceSynchronize();
}
