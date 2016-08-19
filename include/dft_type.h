/*
 * Provides a class interface to DFT types
 *
 * Wither cuFFT (device baseed) or FFTW (host based)
 *
 */


#ifndef DFT_TYPE_H
#define DFT_TYPE_H

// CUDACC has a quirk edit fftw3:
// https://github.com/FFTW/fftw3/issues/18
#include <fftw3.h>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include "cucmplx.h"
#include "cuda_types.h"
#include "error.h"

namespace cufft
{
    // Inline this template function and its specialization as to avoid having multiple definitions of the
    // function when linking:
    //
    // /tmp/tmpxft_00000f3d_00000000-31_slab_bc.o: In function `void plan_dft<double>(int&, int&, twodads::dft_t, twodads::slab_layout_t, cufftResult_t&, double)':
    // /home/rku000/source/2dads/include/dft_type.h:54: multiple definition of `void plan_dft<double>(int&, int&, twodads::dft_t, twodads::slab_layout_t, cufftResult_t&, double)'
    // /tmp/tmpxft_00000f3d_00000000-22_test_derivs.o:/home/rku000/source/2dads/include/dft_type.h:54: first defined here
    //
    // http://stackoverflow.com/questions/4445654/multiple-definition-of-template-specialization-when-using-different-objects#4445772
    template <typename T>
        inline void plan_dft(cufftHandle& plan_r2c, 
                             cufftHandle& plan_c2r, 
                             const twodads::dft_t dft_type, 
                             const twodads::slab_layout_t geom, const T dummy)
        {
            // Do nothing; the constructor should call a template specialization for T=float,double below
        }

    template <>
    inline void plan_dft<double> (cufftHandle& plan_r2c, 
                cufftHandle& plan_c2r, 
                const twodads::dft_t dft_type, 
                const twodads::slab_layout_t geom, 
                const double dummy)
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
        inline void call_dft_r2c(const cufftHandle plan_r2c, T* arr_in, CuCmplx<T>* arr_out) 
        {
            // Do nothing
        }


    template<>
        inline void call_dft_r2c<double>(const cufftHandle plan_r2c, double* arr_in, CuCmplx<double>* arr_out) 
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
        inline void call_dft_r2c<float>(const cufftHandle plan_r2c, float* arr_in, CuCmplx<float>* arr_out) 
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
        inline void call_dft_c2r(const cufftHandle plan_c2r, CuCmplx<T>* in_arr, T* out_arr) 
        {
            // Do nothing
        }

    template<>
        inline void call_dft_c2r<double>(const cufftHandle plan_c2r, CuCmplx<double>* arr_in, double* arr_out) 
        {
            cufftResult err;
            if((err = cufftExecZ2D(plan_c2r, (cufftDoubleComplex*) arr_in, arr_out)))
            {
                std::stringstream err_str;
                err_str << "Error executing cufftZ2D DFT: " << cuda::cufftGetErrorString.at(err) << std::endl;
                throw gpu_error(err_str.str());
            }
        }


    template<>
        inline void call_dft_c2r<float>(const cufftHandle plan_c2r, CuCmplx<float>* arr_in, float* arr_out) 
        {
            cufftResult err;
            if((err = cufftExecC2R(plan_c2r, (cufftComplex*) arr_in, arr_out)))
            {
                std::stringstream err_str;
                err_str << "Error executing cufftC2R DFT: " << cuda::cufftGetErrorString.at(err) << std::endl;
                throw gpu_error(err_str.str());
            }
        }

}


#endif //__CUDACC__

namespace fftw
{
    template <typename T>
        inline void plan_dft(fftw_plan& plan_r2c, fftw_plan& plan_c2r, const twodads::dft_t dft_type,
                const twodads::slab_layout_t geom, const T dummy)
        {
            // Do nothing, the constructor should call a template specialization for T = float, double below
        }

    template <>
        inline void plan_dft<double>(fftw_plan& plan_r2c, fftw_plan& plan_c2r,
                const twodads::dft_t dft_type, const twodads::slab_layout_t geom,
                const double dummy)
        {
            int rank{1};
            int n[]{static_cast<int>(geom.get_my())};
            int howmany{static_cast<int>(geom.get_nx())};
            int idist = static_cast<int>(geom.get_my() + geom.get_pad_y());
            int odist = static_cast<int>(geom.get_my() / 2 + 1);
            int istride{1};
            int ostride{1};

            // Create dummy arrays for FFTW_ESTIMATE
            // We are going to use the new_array execute functions of fftw later on.
            // Here we are planning an in-place dft.
            // Later-on, we may not use in-place DFTs but out of place.
            // For the derivatives, there was a bug when using a planned
            // out-of-place transform with in-place data. 
            // Maybe we need to later plan both, in- and out-of-place DFTs
            // and dispatch according to the pointer addresses passed to dft_r2c/dft_c2r
            double* dummy_double = new double[(geom.get_nx() + geom.get_pad_x()) * (geom.get_my() + geom.get_pad_y())];
            CuCmplx<double>* dummy_cmplx = new CuCmplx<double>[(geom.get_nx() + geom.get_pad_x()) * ((geom.get_my() + geom.get_pad_y()) / 2)];

            switch(dft_type)
            {
                case twodads::dft_t::dft_1d:
                    // Initialize a 1d transformation
                    plan_r2c = fftw_plan_many_dft_r2c(rank,         //int rank
                                                      n,            // Length of transformation (My)
                                                      howmany,      // = Nx
                                                      dummy_double, // double* in (dummy ptr)
                                                      NULL,         // inembed
                                                      istride,      // istride
                                                      idist,        // idist
                                                      reinterpret_cast<fftw_complex*>(dummy_double),  // fftw_complex* out
                                                      NULL,
                                                      ostride,      // ostride
                                                      odist,
                                                      FFTW_ESTIMATE);
                    plan_c2r = fftw_plan_many_dft_c2r(rank,
                                                      n,
                                                      howmany,
                                                      reinterpret_cast<fftw_complex*>(dummy_double),
                                                      NULL,
                                                      ostride,
                                                      odist,
                                                      dummy_double,
                                                      NULL,
                                                      istride,
                                                      idist,
                                                      FFTW_ESTIMATE);
                    break;

                case twodads::dft_t::dft_2d:
                    plan_r2c = fftw_plan_dft_r2c_2d(geom.get_nx(), geom.get_my(), dummy_double, reinterpret_cast<fftw_complex*>(dummy_cmplx), FFTW_ESTIMATE);
                    plan_c2r = fftw_plan_dft_c2r_2d(geom.get_nx(), geom.get_my(), reinterpret_cast<fftw_complex*>(dummy_cmplx), dummy_double, FFTW_ESTIMATE);
                    break;
            }
            delete [] dummy_double;
            delete [] dummy_cmplx;
        }

    template <>
        inline void plan_dft<float>(fftw_plan& plan_r2c, fftw_plan& plan_c2r,
                const twodads::dft_t dft_type, const twodads::slab_layout_t geom,
                const float dummy)
        {
        }

    template <typename T>
        inline void call_dft_r2c(fftw_plan& plan_r2c, T* arr_in, CuCmplx<T>* arr_out)
        {
            // Do nothing but se to that a specialization is called
        }

    template <>
        inline void call_dft_r2c<double>(fftw_plan& plan_r2c, double* arr_in, CuCmplx<double>* arr_out)
        {
            fftw_execute_dft_r2c(plan_r2c, arr_in, reinterpret_cast<fftw_complex*>(arr_out));
        }

    template <>
        inline void call_dft_r2c<float>(fftw_plan& plan_r2c, float* arr_in, CuCmplx<float>* arr_out)
        {
        }

    template <typename T>
        inline void call_dft_c2r(fftw_plan& plan_c2r, CuCmplx<T>* arr_in, T* arr_out)
        {
            // Do nothing
        }

    template <>
        inline void call_dft_c2r<double>(fftw_plan& plan_c2r, CuCmplx<double>* arr_in, double* arr_out)
        {
            fftw_execute_dft_c2r(plan_c2r, reinterpret_cast<fftw_complex*>(arr_in), arr_out);
        }

    template <>
        inline void call_dft_c2r<float>(fftw_plan& plan_c2r, CuCmplx<float>* arr_in, float* arr_out)
        {
        }
}

template <typename T>
class dft_object_t
{
    public:
        virtual ~dft_object_t() {};
        virtual void dft_r2c(T*, twodads::cmplx_t*) = 0;
        virtual void dft_c2r(twodads::cmplx_t*, T*) = 0;
};


#ifdef __CUDACC__
template <typename T>
class cufft_object_t : public dft_object_t<T>
{
    public:
        cufft_object_t(const twodads::slab_layout_t _geom, const twodads::dft_t _dft_type) :geom(_geom), dft_type(_dft_type)
        {
            cufft :: plan_dft(plan_r2c, plan_c2r, get_dft_t(), get_geom(), T{});
        }
        ~cufft_object_t()
        {
            cufftDestroy(plan_r2c);
            cufftDestroy(plan_c2r);
        }

        void dft_r2c(T* arr_in, twodads::cmplx_t* arr_out)
        {
            cufft :: call_dft_r2c(get_plan_r2c(), arr_in, arr_out);
        }
        void dft_c2r(twodads::cmplx_t* arr_in, T* arr_out)
        {
            cufft :: call_dft_c2r(get_plan_c2r(), arr_in, arr_out);
        }

        //inline twodads::slab_layout_t get_geom() const {return(geom);};
        cufftHandle get_plan_r2c() const {return(plan_r2c);};
        cufftHandle get_plan_c2r() const {return(plan_c2r);};
        cufftHandle& get_plan_r2c() {return(plan_r2c);};
        cufftHandle& get_plan_c2r() {return(plan_c2r);};

        inline twodads::slab_layout_t get_geom() const {return(geom);}
        inline twodads::dft_t get_dft_t() const {return(dft_type);}
    private:
        const twodads::slab_layout_t geom;
        const twodads::dft_t dft_type;
        cufftHandle plan_r2c;
        cufftHandle plan_c2r;
};
#endif //__CUDACC__

template <typename T>
class fftw_object_t : public dft_object_t<T>
{
    public:
        fftw_object_t(const twodads::slab_layout_t _geom, const twodads::dft_t _dft_type) :  geom(_geom), dft_type(_dft_type)
        {
            fftw :: plan_dft(plan_r2c, plan_c2r, get_dft_t(), get_geom(), T{});
        }

        ~fftw_object_t()
        {
            fftw_destroy_plan(get_plan_c2r());
            fftw_destroy_plan(get_plan_r2c());
        }

        void dft_r2c(T* arr_in, twodads::cmplx_t* arr_out)
        {
            fftw :: call_dft_r2c(get_plan_r2c(), arr_in, arr_out);
        }

        void dft_c2r(twodads::cmplx_t* arr_in, T* arr_out)
        {
            fftw :: call_dft_c2r(get_plan_c2r(), arr_in, arr_out);
        }

        fftw_plan get_plan_r2c() const {return(plan_r2c);};
        fftw_plan get_plan_c2r() const {return(plan_c2r);};

        fftw_plan& get_plan_r2c() {return(plan_r2c);};
        fftw_plan& get_plan_c2r() {return(plan_c2r);};

        inline twodads::slab_layout_t get_geom() const {return(geom);}
        inline twodads::dft_t get_dft_t() const {return(dft_type);}
    private:
        const twodads::slab_layout_t geom;
        const twodads::dft_t dft_type;
        fftw_plan plan_r2c;
        fftw_plan plan_c2r;
};

#endif //DFT_TYPE_H