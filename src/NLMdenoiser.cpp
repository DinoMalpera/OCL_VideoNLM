#include "NLMdenoiser.h"
#include "Pixel_Value.h"
#include "FrameSize.h"
#include "NLMparams.h"
#include "FrameSequence.h"

using namespace VNLM;

namespace {

bool
verify_radius(
        const unsigned int radius,
        const FrameSize&   frameSize )
{
    if ( 0U == radius )
    {
        return false;
    }

    if ( radius >= frameSize.size_x )
    {
        return false;
    }

    if ( radius >= frameSize.size_y )
    {
        return false;
    }

    return true;
}

bool
verify_params(
        const NLMparams& params,
        const FrameSize  frameSize )
{
    if ( false == verify_radius( params.patch_radius, frameSize ) )
    {
        return false;
    }

    if ( false == verify_radius( params.search_window_radius, frameSize ) )
    {
        return false;
    }

    if ( 0.0 > params.standard_deviation_of_noise )
    {
        return false;
    }

    if ( 0.0 >= params.filtering_parameter )
    {
        return false;
    }

    return true;
}

}

template <ComputableColor Pixel_Value_policy>
bool
NLMdenoiser::verify(
        const   FrameSequence<Pixel_Value_policy>&  frameSequence,
        const   NLMparams&                          params )
{
    if ( false == frameSequence.verify() )
    {
        return false;
    }

    if ( false == verify_params( params, frameSequence.getFrameSize() ) )
    {
        return false;
    }

    return true;
}

template <ComputableColor Pixel_Value_policy>
void
NLMdenoiser::Denoise(
        const   FrameSequence<Pixel_Value_policy>&  frameSequence,
        Frame<Pixel_Value_policy>&                  result,
        const NLMparams&                            params )
{
    if ( false == verify( frameSequence, params ) )
    {
        return;
    }

    NLMVideo_OpenCL( frameSequence, result, params );
}


// explicit template initializations for ColorSpace parameter

// RGB
        
template
bool
NLMdenoiser::verify<Color_Space_RGB>(
        const FrameSequence<Color_Space_RGB>&,
        const NLMparams& );
template
void
NLMdenoiser::Denoise<Color_Space_RGB>(
        const FrameSequence<Color_Space_RGB>&,
        Frame<Color_Space_RGB>&,
        const NLMparams& );
        

