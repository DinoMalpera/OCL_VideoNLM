#pragma once

#include "../NLMcommon.h"
#include "../Pixel_Coord.h"

namespace VNLM
{

    /*  Pixel iterator.
     *  Visits all pixels in row-first order.
     * */
    class Pixel_Range_Iterator
    {
    public:
        constexpr
        Pixel_Range_Iterator(
                const unsigned int  x,
                const unsigned int  y,
                const unsigned int  x_size
        ) noexcept
            :   x       (x)
            ,   y       (y)
            ,   x_size  (x_size)
        {
            VNLM_ASSERT( x <= x_size )
        }
    public:
        constexpr
        Pixel_Range_Iterator(
                const Pixel_Coord   pc,
                const unsigned int  x_size
        ) noexcept
            :   x       (pc.x)
            ,   y       (pc.y)
            ,   x_size  (x_size)
        {
            VNLM_ASSERT( x <= x_size )
        }
    public:
        constexpr
        Pixel_Range_Iterator&
        operator*() noexcept
        {
            return *this;
        }
    public:
        constexpr
        Pixel_Range_Iterator&
        operator++() noexcept
        {
            ++x;
            if ( x_size <= x ) [[unlikely]]
            {
                ++y;
                x = 0U;
            }
            
            return *this;
        }
    public:
        constexpr
        bool
        operator!=(
                const Pixel_Range_Iterator& _end ) const noexcept
        {
            VNLM_ASSERT( x_size == _end.x_size )
            return ( _end.y != y );
        }
    public:
        constexpr
        Pixel_Range_Iterator
        operator+(
                const unsigned int off ) const noexcept
        {
            const unsigned int yy = off / x_size;
            const unsigned int xx = off - yy * x_size;
            return { xx, yy, x_size };
        }
    public:
        constexpr
        unsigned int
        getX() const noexcept
        {
            return x;
        }
    public:
        constexpr
        unsigned int
        getY() const noexcept
        {
            return y;
        }
    public:
        constexpr
        Pixel_Coord
        getPixelCoord() const noexcept
        {
            return { x, y };
        }
    public:
        constexpr 
        unsigned int
        get_x_size() const noexcept
        {
            return x_size;
        }
        
    private:
        unsigned int    x;
        unsigned int    y;
        unsigned int    x_size;
    };

}
