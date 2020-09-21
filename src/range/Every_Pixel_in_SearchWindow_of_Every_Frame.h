#pragma once

#include "../NLMcommon.h"
#include "Pixel_Range_Iterator.h"
#include "../FrameSize.h"

namespace VNLM
{

    /*  Iterates over every pixel in the search window.
    *
    *   Scales the window down when it crosses the border.
    */
    class Every_Pixel_in_SearchWindow_of_Every_Frame
    {
    public:
        class iterator
        {
        public:
            constexpr
            iterator(
                    const Pixel_Range_Iterator  frame_pixel_iterator,
                    const Pixel_Range_Iterator  frame_pixel_iterator_end,
                    const Pixel_Coord           beginning_of_window,
                    const unsigned int          frame_index
            ) noexcept
                :   frame_pixel_iterator        (frame_pixel_iterator)
                ,   frame_pixel_iterator_end    (frame_pixel_iterator_end)
                ,   beginning_of_window         (beginning_of_window)
                ,   frame_index                 (frame_index)
            {
                VNLM_ASSERT( frame_pixel_iterator.get_x_size() == frame_pixel_iterator_end.get_x_size() )
            }
        public:
            constexpr
            iterator&
            operator*() noexcept
            {
                return *this;
            }
        public:
            constexpr
            iterator&
            operator++() noexcept
            {
                ++frame_pixel_iterator;
                if ( reached_end_of_search_window_for_current_frame() ) [[unlikely]]
                {
                    ++frame_index;
                    reset_frame_pixel_iterator();
                }
                
                return *this;
            }
        public:
            constexpr
            bool
            operator!=(
                    const iterator& _end ) const noexcept
            {
                return  (frame_index != _end.frame_index);
            }
        public:
            Pixel_Coord
            getPixelCoord() const noexcept
            {
                return frame_pixel_iterator.getPixelCoord() + beginning_of_window;
            }
        public:
            unsigned int
            getFrame() const noexcept
            {
                return frame_index;
            }
            
        private:
            constexpr
            bool
            reached_end_of_search_window_for_current_frame() const noexcept
            {
                const bool fpi = (frame_pixel_iterator != frame_pixel_iterator_end);
                return false == fpi;
            }
        private:
            constexpr
            void
            reset_frame_pixel_iterator() noexcept
            {
                frame_pixel_iterator = Pixel_Range_Iterator{ 0, 0, frame_pixel_iterator.get_x_size() };
            }
            
        private:
                    Pixel_Range_Iterator    frame_pixel_iterator;
            const   Pixel_Range_Iterator    frame_pixel_iterator_end;
            const   Pixel_Coord             beginning_of_window;
                    unsigned int            frame_index;
        };
    public:
        Every_Pixel_in_SearchWindow_of_Every_Frame(
                const FrameSize&    frameSize,
                const unsigned int  sequence_length,
                const unsigned int  search_window_radius,
                const Pixel_Coord   window_center
        ) noexcept
            :   sequence_length         (sequence_length)
            ,   search_window_size_x    (compute_search_window_size( search_window_radius, window_center.x, frameSize.size_x ))
            ,   search_window_size_y    (compute_search_window_size( search_window_radius, window_center.y, frameSize.size_y ))
            ,   beginning_of_window     (compute_offset_to_the_beginning_of_window( search_window_radius, window_center ))
        {
        }
    public:
        constexpr
        iterator
        begin() const noexcept
        {
            return {
                .frame_pixel_iterator     = Pixel_Range_Iterator{ 0, 0, search_window_size_x },
                .frame_pixel_iterator_end = Pixel_Range_Iterator{ 0, search_window_size_y, search_window_size_x },
                .beginning_of_window      = beginning_of_window,
                .frame_index              = 0 };
        }
    public:
        constexpr
        iterator
        end() const noexcept
        {
            const Pixel_Range_Iterator pir{
                0,
                search_window_size_y,
                search_window_size_x };
            
            return {
                .frame_pixel_iterator     = pir,
                .frame_pixel_iterator_end = pir,
                .beginning_of_window      = beginning_of_window,
                .frame_index              = sequence_length };
        }
        
    private:
        /*  Computes the search window size, accounting for the frame border.
        */
        static 
        unsigned int
        compute_search_window_size(
                const unsigned int search_window_radius,
                const unsigned int window_center,
                const unsigned int frameSize ) noexcept
        {
            unsigned int search_window_size = 2U * search_window_radius + 1U;
            
            if ( window_center < search_window_radius )
            {
                search_window_size -= search_window_radius - window_center;
            }
            
            if ( const unsigned int distance_to_upper_border = frameSize - window_center - 1U;
                 distance_to_upper_border < search_window_radius )
            {
                search_window_size -= search_window_radius - distance_to_upper_border;
            }
            
            return search_window_size; 
        }
    private:
        static 
        Pixel_Coord
        compute_offset_to_the_beginning_of_window(
                const unsigned int  search_window_radius,
                const Pixel_Coord   window_center ) noexcept
        {
            return {
                .x = ( window_center.x >= search_window_radius ) ? (window_center.x - search_window_radius) : 0U,
                .y = ( window_center.y >= search_window_radius ) ? (window_center.y - search_window_radius) : 0U };
        }
    
    private:
        const unsigned int  sequence_length;
        const unsigned int  search_window_size_x;
        const unsigned int  search_window_size_y;
        const Pixel_Coord   beginning_of_window;
    };

}

