/*
 * Provides a class interface to DFT types
 *
 * Wither cuFFT (device baseed) or FFTW (host based)
 *
 */


#ifndef DFT_TYPE_H
#define DFT_TYPE_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include "cuda_types.h"
#include "error.h"


// Inline this template function and its specialization as to avoid having multiple definitions of the
// function when linking:
//
// /tmp/tmpxft_00000f3d_00000000-31_slab_bc.o: In function `void plan_dft<double>(int&, int&, twodads::dft_t, twodads::slab_layout_t, cufftResult_t&, double)':
// /home/rku000/source/2dads/include/dft_type.h:54: multiple definition of `void plan_dft<double>(int&, int&, twodads::dft_t, twodads::slab_layout_t, cufftResult_t&, double)'
// /tmp/tmpxft_00000f3d_00000000-22_test_derivs.o:/home/rku000/source/2dads/include/dft_type.h:54: first defined here
//
// http://stackoverflow.com/questions/4445654/multiple-definition-of-template-specialization-when-using-different-objects#4445772
template <typename T>
inline void plan_dft(twodads::fft_handle_t& plan_r2c, twodads::fft_handle_t& plan_c2r, const twodads::dft_t dft_type, const twodads::slab_layout_t geom, const T dummy)
{
    // Do nothing; the constructor should call a template specialization for T=float,double below
}

template <>
inline void plan_dft<double> (twodads::fft_handle_t& plan_r2c, twodads::fft_handle_t& plan_c2r, const twodads::dft_t dft_type, const twodads::slab_layout_t geom, const double dummy)
{
	cufftResult err;
    int dft_size[2] = {0, 0};       // Size of the transformation
    int dft_onembed[2] = {0, 0};    // Input, embedded size of the transformation
    //int dft_inembed[2] = {0, 0};    // Output, embedded size of the transformation
    int dist_real{0};               // Distance between two vectors inpu vectors for DFT, in units of double
    int dist_cplx{0};               // Distance between two complex input vectors for iDFT, in units of (2 * double)
    int istride{1};                 // Distance between two successive input and output elements in the least significant (that is, the innermost) dimension
    int ostride{1};
    switch(dft_type)
    {
        case twodads::dft_t::dft_1d:
            // Initialize a 1d transformation
            dft_size[0] = static_cast<int>(geom.get_my());
            dft_onembed[0] = static_cast<int>(geom.get_my() / 2 + 1);
            dist_real = static_cast<int>(geom.get_my() + geom.get_pad_y()); //int(My + geom.pad_y); 
            dist_cplx = static_cast<int>(geom.get_my() / 2 + 1); //int(My / 2 + 1); 

            // Plan the DFT, D2Z
            if ((err = cufftPlanMany(&plan_r2c, 
                                     1,             //int rank
                                     dft_size,      //int* n
                                     dft_size,      //int* inembed
                                     istride,       //int istride
                                     dist_real,     //int idist
                                     dft_onembed,   //int* onembed
                                     ostride,       //int ostride
                                     dist_cplx,     //int odist
                                     CUFFT_D2Z,     //cufftType type
                                     static_cast<int>(geom.get_nx()))            //int batch
                ) != CUFFT_SUCCESS)
            {
                std::stringstream err_str;
                err_str << "Error planning 1d D2Z DFT: " << err << "\n";
                throw gpu_error(err_str.str());
            }
           
            // Plan the iDFT, Z2D 
            if((err = cufftPlanMany(&plan_c2r,
                                    1,              //int rank
                                    dft_size,       //int* n
                                    dft_onembed,    //int* inembed
                                    istride,        //int istride
                                    dist_cplx,      //int idist
                                    dft_size,       //int* onembed
                                    ostride,        //int ostride
                                    dist_real,      //int odist
                                    CUFFT_Z2D,      //cufftType type
                                    static_cast<int>(geom.get_nx()))            //int batch
               ) != CUFFT_SUCCESS)
            {
                std::stringstream err_str;
                err_str << "Error planning 1d Z2D DFT: " << err << "\n";
                throw gpu_error(err_str.str());
            }
            break;

        case twodads::dft_t::dft_2d:
            // Initialize 2d transformation
            dft_size[1] = static_cast<int>(geom.get_my()); //int(My);
            dft_size[0] = static_cast<int>(geom.get_nx()); //int(Nx);
            //dft_inembed[1] = static_cast<int>(geom.get_my() + geom.get_pad_y()); //My + geom.pad_y; 
            //dft_onembed[1] = static_cast<int>(geom.get_my() / 2 + 1); //My / 2 + 1;
            istride = 1;
            ostride = 1;

            // Plan 2d r2c transformation
            //if((err = cufftPlanMany(&plan_r2c,
            //                        2,              //int rank
            //                        dft_size,       //int* n
            //                        dft_inembed,    //int* inembed
            //                        istride,        //int istride
            //                        (Nx + geom.pad_y) * (My + geom.pad_y), //int idist
            //                        dft_onembed,    //int* onembed
            //                        ostride,        //int ostride
            //                        Nx * (My / 2 + 1), //int odist
            //                        CUFFT_D2Z,      //cufftType typ
            //                        1)              //int batch
            //   ) != CUFFT_SUCCESS)
            //if((err = cufftPlan2d(&plan_r2c, Nx, My, CUFFT_D2Z) )!= CUFFT_SUCCESS)
            if((err = cufftPlan2d(&plan_r2c, geom.get_nx(), geom.get_my(), CUFFT_D2Z) )!= CUFFT_SUCCESS)
            {
                std::stringstream err_str;
                err_str << "Error planning 2d D2Z DFT: " << err << "\n";
                throw gpu_error(err_str.str());
            }

            // Plan 2d c2r transformation
            //if((err = cufftPlanMany(&plan_c2r,
            //                        2,
            //                        dft_size,
            //                        dft_onembed,
            //                        ostride,
            //                        Nx * (My / 2 + 1),
            //                        dft_inembed,
            //                        istride,
            //                        Nx * (My + geom.pad_y),
            //                        CUFFT_Z2D,
            //                        1)
            //  ) != CUFFT_SUCCESS)
            //if((err = cufftPlan2d(&plan_c2r, Nx, My, CUFFT_Z2D) )!= CUFFT_SUCCESS)
            if((err = cufftPlan2d(&plan_c2r, geom.get_nx(), geom.get_my(), CUFFT_Z2D) )!= CUFFT_SUCCESS)
            {
                std::stringstream err_str;
                err_str << "Error planning 2d Z2D DFT: " << err << "\n";
                throw gpu_error(err_str.str());
            }
            break;
    }
}



// Generic template for c2r functions
template <typename T>
inline void call_dft_r2c(twodads::fft_handle_t plan_r2c, T* arr_in, twodads::cmplx_t* arr_out) 
{
    // Do nothing
}


template<>
inline void call_dft_r2c<double>(twodads::fft_handle_t plan_r2c, double* arr_in, twodads::cmplx_t* arr_out) 
{
    cufftResult err;
    if((err = cufftExecD2Z(plan_r2c, arr_in, (cufftDoubleComplex*) arr_out)))
    {
        std::stringstream err_str;
        err_str << "Error executing cufftD2Z DFT: " << cuda::cufftGetErrorString.at(err) << std::endl;
        throw gpu_error(err_str.str());
    }
}


template<>
inline void call_dft_r2c<float>(twodads::fft_handle_t plan_r2c, float* arr_in, twodads::cmplx_t* arr_out) 
{
    cufftResult err;
    if((err = cufftExecR2C(plan_r2c, arr_in, (cufftComplex*) arr_out)))
    {
        std::stringstream err_str;
        err_str << "Error executing cufftD2Z DFT: " << cuda::cufftGetErrorString.at(err) << std::endl;
        throw gpu_error(err_str.str());
    }
}


template <typename T>
inline void call_dft_c2r(twodads::fft_handle_t plan_c2r, twodads::cmplx_t* in_arr, T* out_arr) 
{
    // Do nothing
}

template<>
inline void call_dft_c2r<double>(twodads::fft_handle_t plan_c2r, twodads::cmplx_t* arr_in, double* arr_out) 
{
    cufftResult err;
    if((err = cufftExecZ2D(plan_c2r, (cufftDoubleComplex*) arr_in, arr_out)))
    {
        std::stringstream err_str;
        err_str << "Error executing cufftZ2D DFT: " << cuda::cufftGetErrorString.at(err) << std::endl;
        throw gpu_error(err_str.str());
    }
    std::cerr << "!!_!_!_!_!1-1-1--1-1--1!!!!! DFT_C2R" << std::endl;
}


template<>
inline void call_dft_c2r<float>(twodads::fft_handle_t plan_c2r, twodads::cmplx_t* arr_in, float* arr_out) 
{
    cufftResult err;
    if((err = cufftExecC2R(plan_c2r, (cufftComplex*) arr_in, arr_out)))
    {
        std::stringstream err_str;
        err_str << "Error executing cufftC2R DFT: " << cuda::cufftGetErrorString.at(err) << std::endl;
        throw gpu_error(err_str.str());
    }
}


template <typename T>
class dft_object_t
{
    public:
        virtual ~dft_object_t() {};
        virtual void dft_r2c(T*, twodads::cmplx_t*) = 0;
        virtual void dft_c2r(twodads::cmplx_t*, T*) = 0;

        //virtual twodads::fft_handle_t get_plan_r2c();
        //virtual twodads::fft_handle_t get_plan_c2r();
};


template <typename T>
class cufft_object_t : public dft_object_t<T>
{
    public:
        cufft_object_t(const twodads::slab_layout_t, const twodads::dft_t);
        ~cufft_object_t();

        void dft_r2c(T*, twodads::cmplx_t*);
        void dft_c2r(twodads::cmplx_t*, T*);

        //inline twodads::slab_layout_t get_geom() const {return(geom);};
        twodads::fft_handle_t get_plan_r2c() const {return(plan_r2c);};
        twodads::fft_handle_t get_plan_c2r() const {return(plan_c2r);};

        inline twodads::slab_layout_t get_geom() const {return(geom);}
        inline twodads::dft_t get_dft_t() const {return(dft_type);}
    private:
        const twodads::slab_layout_t geom;
        const twodads::dft_t dft_type;
        twodads::fft_handle_t plan_r2c;
        twodads::fft_handle_t plan_c2r;
};


template <typename T>
cufft_object_t<T> :: cufft_object_t(const twodads::slab_layout_t _geom, const twodads::dft_t _dft_type) : 
    geom(_geom), dft_type(_dft_type)
{
    // Create a dummy instance of T so that template specialization works out.
    T dummy{0.0};
    plan_dft(plan_r2c, plan_c2r, get_dft_t(), get_geom(), dummy);
}


template <typename T>
inline void cufft_object_t<T> :: dft_r2c(T* arr_in, twodads::cmplx_t* arr_out) 
{
    call_dft_r2c(get_plan_r2c(), arr_in, arr_out);
}


template <typename T>
inline void cufft_object_t<T> :: dft_c2r(twodads::cmplx_t* arr_in, T* arr_out) 
{
    call_dft_c2r(get_plan_c2r(), arr_in, arr_out);
}


template <typename T>
cufft_object_t<T> :: ~cufft_object_t()
{
    std::cout << "~cufft_object()" << std::endl;
    cufftDestroy(plan_r2c);
    cufftDestroy(plan_c2r);
}

#endif //DFT_TYPE_H
