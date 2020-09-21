#include <vector>
#include <string>
#include <iostream>
#include "NLMdenoiser.h"
#include "FrameSize.h"
#include "FrameSequence.h"
#include "NLMparams.h"
#include "Pixel_Value.h"
#include "range/Every_Pixel_In_a_Frame.h"
#include "OpenCL_utils.h"

using namespace VNLM;

namespace {

void
set_ocl_by_pixelValue(
        cl_float3&             lhs,
        const Color_Space_RGB& rhs ) noexcept
{
    lhs.s[0] = rhs.r;
    lhs.s[1] = rhs.g;
    lhs.s[2] = rhs.b;
}

void
set_pixelValue_by_ocl(
        Color_Space_RGB& lhs,
        const cl_float3& rhs ) noexcept
{
    lhs.r = rhs.s[0];
    lhs.g = rhs.s[1];
    lhs.b = rhs.s[2];
}

unsigned int
handle_border(
        const unsigned int i,
        const unsigned int patch_radius,
        const unsigned int frame_end,
        const unsigned int fac ) noexcept
{
    unsigned int ii = i - patch_radius;

    if ( i < patch_radius )
    {
        ii = patch_radius - i;
    }
    if ( i >= frame_end )
    {
        ii = fac - i;
    }
    
    return ii;
}

/// We want to eliminate branches from OpenCL code, so we are
/// extending frame by the patch radius so that the border cases
/// don't have to be handeled in the kernel.
template <ComputableColor Pixel_Value_policy>
std::vector<cl_float3>
extend_frame_by_patch_radius(
        const Pixel_Value_policy*const image,
        const FrameSize&               frame_size,
        const unsigned int             patch_radius )
{
    const unsigned int patch_radius_2 = 2u * patch_radius;
    const unsigned int frame_end_x    = frame_size.size_x + patch_radius;
    const unsigned int frame_end_y    = frame_size.size_y + patch_radius;
    const unsigned int fac_x          = 2u*frame_size.size_x + patch_radius - 2u;
    const unsigned int fac_y          = 2u*frame_size.size_y + patch_radius - 2u;
    const unsigned int range_end_x    = frame_size.size_x + patch_radius_2;
    const unsigned int range_end_y    = frame_size.size_y + patch_radius_2;
    
    std::vector<cl_float3> extended_img( range_end_x * range_end_y );
    
    const unsigned int nx = frame_size.size_x + patch_radius_2;
    
    for( unsigned int j=0u; j<range_end_y; ++j )
    {
        const unsigned int jj =
            handle_border(
                j,
                patch_radius,
                frame_end_y,
                fac_y );
                
        for( unsigned int i=0u; i<range_end_x; ++i )
        {
            const unsigned int ii =
                handle_border(
                    i,
                    patch_radius,
                    frame_end_x,
                    fac_x );
                    
            set_ocl_by_pixelValue(
                extended_img[j*nx + i],
                image[jj*frame_size.size_x + ii] );
        }
    }
    
    return extended_img;
}

auto
setup_ocl(
        const std::string& kernel_source_file,
        const char*const   kernel_function )
{
    cl::Device       device = get_first_available_device();
    
    cl::Context      context( device );
    cl::CommandQueue queue( context, device );
    
    cl::Program      program(
        context,
        load_kernel_source( kernel_source_file ),
        false );
            
    try
    {
        program.build( std::vector<cl::Device>{device} );
    }
    catch (const cl::Error&)
    {
        std::cerr << "Compilation Failed!" << std::endl;
        std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
        std::cerr << std::endl;
        throw 2;
    }
    
    cl::Kernel nlmVideo_kernel( program, kernel_function );
        
    return std::tuple( std::move(context), std::move(queue), std::move(nlmVideo_kernel) );
}

template <ComputableColor Pixel_Value_policy>
auto
transform_frame_sequence(
        const FrameSequence<Pixel_Value_policy>&  frameSequence,
        const unsigned int patch_radius )
{
    std::vector<std::vector<cl_float3>> frameSequence_buffer;
    
    const auto frame_size = frameSequence.getFrameSize();
    const unsigned int number_of_frames = frameSequence.get_sequence_size();
    
    for( unsigned int frame_in_sequence=0u; frame_in_sequence<number_of_frames; ++frame_in_sequence )
    {
        frameSequence_buffer.push_back(
            extend_frame_by_patch_radius(
                frameSequence.get_frame(frame_in_sequence).get_data_view(),
                frame_size,
                patch_radius ) );
    }
    
    return frameSequence_buffer;
}

template <ComputableColor Pixel_Value_policy>
auto
transform_result(
        const FrameSize&              frame_size,
        const std::vector<cl_float4>& result_b,
        Frame<Pixel_Value_policy>&    result )
{
    const Every_Pixel_In_a_Frame result_iterator( frame_size );
    
    auto it_r           = result_iterator.begin();
    const auto it_r_end = result_iterator.end();
    auto it_b           = result_b.begin();
    
    for(  ; it_r!=it_r_end ; ++it_r, ++it_b )
    {
        assert( it_b != result_b.end() );
        auto& rez_value = result[it_r.getPixelCoord()];
        auto  buf_value = *it_b;
        const float i_weight = 1.0f / buf_value.s[3u];
        
        buf_value.s[0] *= i_weight;
        buf_value.s[1] *= i_weight;
        buf_value.s[2] *= i_weight;
        
        set_pixelValue_by_ocl( rez_value, buf_value );
    }
    assert( it_b == result_b.end() );
}

}

template <ComputableColor Pixel_Value_policy>
void
NLMdenoiser::NLMVideo_OpenCL(
        const   FrameSequence<Pixel_Value_policy>&  frameSequence,
                Frame<Pixel_Value_policy>&          result,
        const   NLMparams&                          params )
{
    try
    {
        auto [ context, queue, nlmVideo_kernel ] = setup_ocl( "kernels/nlm_kernel.cl", "compute_search_window" );
        
        const auto frame_size = frameSequence.getFrameSize();
        
        std::vector<std::vector<cl_float3>> frameSequence_buffer
            = transform_frame_sequence( frameSequence, params.patch_radius );
        
        std::vector<cl_float4> result_b( frame_size.size_x * frame_size.size_y );
        
        const unsigned int central_frame = frameSequence.get_center_frame_index();
        
        cl::Buffer central_cl(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            frameSequence_buffer[central_frame].size() * sizeof(cl_float3),
            frameSequence_buffer[central_frame].data());
        cl::Buffer result_cl(
            context,
            CL_MEM_READ_WRITE,
            result_b.size() * sizeof(cl_float4) );
            
        const unsigned int number_of_frames = frameSequence.get_sequence_size();
            
        for( unsigned int current_frame_ix=0u; current_frame_ix<number_of_frames; ++current_frame_ix )
        {
            cl::Buffer all_cl(
                context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                frameSequence_buffer[current_frame_ix].size() * sizeof(cl_float3),
                frameSequence_buffer[current_frame_ix].data());

            unsigned int arg_cntr = 0u;

            nlmVideo_kernel.setArg( arg_cntr++, central_cl );
            nlmVideo_kernel.setArg( arg_cntr++, all_cl );
            nlmVideo_kernel.setArg( arg_cntr++, result_cl );
            nlmVideo_kernel.setArg( arg_cntr++, static_cast<cl_int>( params.search_window_radius ) );
            nlmVideo_kernel.setArg( arg_cntr++, static_cast<cl_int>( params.patch_radius ) );
            nlmVideo_kernel.setArg( arg_cntr++, cl_int2{ static_cast<cl_int>(frame_size.size_x), static_cast<cl_int>(frame_size.size_y) } );
            nlmVideo_kernel.setArg( arg_cntr++, static_cast<float>( params.standard_deviation_of_noise ) );
            nlmVideo_kernel.setArg( arg_cntr++, static_cast<float>( params.filtering_parameter ) );

            queue.enqueueNDRangeKernel(
                nlmVideo_kernel,
                cl::NullRange,
                cl::NDRange( frame_size.size_x, frame_size.size_y ),
                cl::NullRange);
        }
            
        queue.enqueueReadBuffer(
            result_cl,
            CL_TRUE,
            0,
            result_b.size() * sizeof(cl_float4),
            result_b.data() );
            
        transform_result( frame_size, result_b, result );
    }
    catch ( const int& e )
    {
        return;
    }
    catch ( const cl::Error& e )
    {
        std::cerr << "Error!" << std::endl;
        std::cerr << e.what() << std::endl;
        std::cerr << e.err() << std::endl;
        std::cerr << std::endl;
        return;
    }
}

template
void
NLMdenoiser::NLMVideo_OpenCL<Color_Space_RGB>(
        const   FrameSequence<Color_Space_RGB>&  frameSequence,
                Frame<Color_Space_RGB>&          result,
        const   NLMparams&                       params );

