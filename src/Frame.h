#pragma once

#include "NLMcommon.h"
#include "Pixel_Coord.h"
#include "FrameSize.h"
#include "Policy_Contracts.h"

namespace VNLM
{

    /*  Frame container. It holds (but not owns) data
     *  for a single image (single frame). Depending on
     *  the pixel_value_type policy colors can be expressed
     *  in different color spaces.
     *  Image pixel data expected to be float [0.0f,1.0f].
     * */
    template <ComputableColor pixel_value_type>
    class Frame
    {
    private:
        using PixelArray        = pixel_value_type*;
    
    // interface
    
    public:
	/*  Frame constructor.
	*   Image pixel (per channel) data expected to be float [0.0f,1.0f].
	* */
        Frame(
                const PixelArray    data,
                const FrameSize&    frameSize,
                const unsigned int  frame
        )
            :   data        (data)
            ,   frameSize   (&frameSize)
            ,   frame       (frame)
        {
            VNLM_ASSERT( nullptr != data )
        }
    public:
        pixel_value_type&
        operator[](
                const Pixel_Coord ix ) noexcept
        {
            const auto pos = compute_index(ix);
            VNLM_ASSERT( pos < (frameSize->size_x*frameSize->size_y) )
            return data[ pos ];
        }
    public:
        const pixel_value_type&
        operator[](
                const Pixel_Coord ix ) const noexcept
        {
            const auto pos = compute_index(ix);
            VNLM_ASSERT( pos < (frameSize->size_x*frameSize->size_y) )
            return data[ pos ];
        }
    public:
        const FrameSize&
        getFrameSize() const noexcept
        {
            return *frameSize;
        }
    public:
        const pixel_value_type* const
        get_data_view() const noexcept
        {
            return data;
        }
        
    // methods
    
    private:
        /* Computes the frame 2D index.
        * */
        constexpr
        size_t
        compute_index(
                const Pixel_Coord& pt ) const noexcept
        {
            VNLM_ASSERT( pt.x < frameSize->size_x )
            return pt.y * frameSize->size_x + pt.x;
        }
        
    // members
    
    private:
                PixelArray      data;
        const   FrameSize*      frameSize;
                unsigned int    frame;
    };

}

