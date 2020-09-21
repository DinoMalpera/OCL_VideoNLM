#pragma once

#include <boost/circular_buffer.hpp>
#include "NLMcommon.h"
#include "Frame.h"
#include "range/Every_Pixel_in_SearchWindow_of_Every_Frame.h"
#include "Policy_Contracts.h"

namespace VNLM
{

    /*  Holds successive frames. One of the frames is the one
     *  we want to denoise (maybe the one in the middle - center frame).
     *  As a frame is added to the sequence
     *  All frames (images) should be of the same size.
     * */
    template <ComputableColor Pixel_Value_policy>
    class FrameSequence
    {
        using FrameArray = boost::circular_buffer<Frame<Pixel_Value_policy>>;
    
    public:
        FrameSequence(
                const unsigned int size_of_sequence
        )
            :   frames  (size_of_sequence)
            ,   center_frame_index  ( size_of_sequence / 2U )
        {}
    public:
        /*  Add frame to the end of the sequence.
        *   If the sequence is full, a frame is removed from the front
        *   of the sequence.
        */
        void
        push_back(
                const Frame<Pixel_Value_policy>& fr )
        {
            VNLM_ASSERT( frames.size()
                         ?
                         (    (fr.getFrameSize().size_x==frames[0].getFrameSize().size_x)
                           && (fr.getFrameSize().size_y==frames[0].getFrameSize().size_y))
                         :
                         true )
            frames.push_back( fr );
        }
    public:
        /*  Get frame from the middle of the sequence.
        *   This will usually be the frame we are denoising.
        */
        const Frame<Pixel_Value_policy>&
        get_center_frame() const noexcept
        {
            VNLM_ASSERT( center_frame_index < frames.size() )
            return frames[center_frame_index];
        }
    public:
        /*  Set which frame from the sequence is considered central.
        *   This might be useful at the beginning or at the end of
        *   the video clip.
        */
        void
        set_center_frame_index(
                const unsigned int _center_frame_index ) noexcept
        {
            center_frame_index = _center_frame_index;
        }
    public:
        unsigned int
        get_center_frame_index() const noexcept
        {
            return center_frame_index;
        }
    public:
        const Frame<Pixel_Value_policy>&
        get_frame(
                const unsigned int ix ) const noexcept
        {
            VNLM_ASSERT( ix < frames.size() )
            return frames[ix];
        }
    public:
        unsigned int
        get_sequence_size() const noexcept
        {
            return frames.size();
        }
    public:
        auto
        operator[](
                const Every_Pixel_in_SearchWindow_of_Every_Frame::iterator& it ) const noexcept
        {
            VNLM_ASSERT( it.getFrame() < frames.size() )
            VNLM_ASSERT( (it.getPixelCoord().y*frames[it.getFrame()].getFrameSize().size_x + it.getPixelCoord().x)
                        <
                        (frames[it.getFrame()].getFrameSize().size_x*frames[it.getFrame()].getFrameSize().size_y) )
            return frames   [ it.getFrame() ]
                            [ it.getPixelCoord() ];
        }
    public:
        FrameSize
        getFrameSize() const noexcept
        {
            VNLM_ASSERT( 0 < frames.size() )
            return frames[0].getFrameSize();
        }
    public:
        /*  Verifies that the frame sequence is consistent.
        */
        bool
        verify() const noexcept
        {
            const unsigned int frameSequence_length = get_sequence_size();
            if ( 0U == frameSequence_length )
            {
                return false;
            }

            const auto center_frame_Index = get_center_frame_index();

            if ( center_frame_Index >= frameSequence_length )
            {
                return false;
            }

            const auto frameSize = getFrameSize();
            if (    (0U == frameSize.size_x)
                 || (0U == frameSize.size_y) )
            {
                return false;
            }

            for( unsigned int frame_ix=0U; frame_ix<frameSequence_length; ++frame_ix )
            {
                const auto _frameSize = get_frame( frame_ix ).getFrameSize();
                if (   (frameSize.size_x != _frameSize.size_x)
                    || (frameSize.size_y != _frameSize.size_y) )
                {
                    return false;
                }
            }

            return true;
        }
    
    private:
        FrameArray      frames;
        unsigned int    center_frame_index;
    };

}

