#pragma once

#include "../FrameSize.h"
#include "../NonCopyable.h"
#include "Pixel_Range_Iterator.h"

namespace VNLM
{

    /*  Iterates over every pixel in the frame.
    */
    class Every_Pixel_In_a_Frame : private NonCopyable
    {
    public:
        Every_Pixel_In_a_Frame(
                const FrameSize& _frameSize
        ) noexcept
            :   frameSize   (_frameSize)
        {
        }
    public:
        constexpr
        Pixel_Range_Iterator
        begin() const noexcept
        {
            return { .x=0U, .y=0U, .x_size=frameSize.size_x };
        }
    public:
        constexpr
        Pixel_Range_Iterator
        end() const noexcept
        {
            return {    .x     = 0U,
                        .y     = frameSize.size_y,
                        .x_size= frameSize.size_x };
        }
    
    private:
        const FrameSize&    frameSize;
    };
}

