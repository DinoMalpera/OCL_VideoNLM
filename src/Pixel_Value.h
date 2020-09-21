#pragma once

#include "NLMutils.h"

namespace VNLM
{

    /*  Grayscale policy
    * */
    using Color_Space_Grayscale = float;
    
    /*  RGB policy
    *
    *   Holds color information and bunch of operators
    *   to make it's interface compliant with float;
    *   NLM algorithm doesn't care what per-pixel data
    *   represents and how it's encoded. Let's keep it
    *   color space agnostic.
    * */
    class Color_Space_RGB
    {
    public:
        Color_Space_RGB() noexcept {}
    public:
        explicit
        Color_Space_RGB(
                const double _
        ) noexcept
            :   r   (_)
            ,   g   (_)
            ,   b   (_)
        {}
    public:
        explicit
        Color_Space_RGB(
                const float r,
                const float g,
                const float b
        ) noexcept
            :   r   (r)
            ,   g   (g)
            ,   b   (b)     
        {}
    public:
        Color_Space_RGB
        operator*(
                const double d ) const noexcept
        {
            return Color_Space_RGB( r*d, g*d, b*d );
        }
    public: 
        Color_Space_RGB&
        operator+=(
                const Color_Space_RGB rhs ) noexcept
        {
            r += rhs.r;
            g += rhs.g;
            b += rhs.b;
        
            return *this;
        }
    public:
        Color_Space_RGB
        operator/(
                const double d ) const noexcept
        {
            return Color_Space_RGB( r/d, g/d, b/d );
        }
    public:
        Color_Space_RGB
        operator-(
                const Color_Space_RGB& rhs ) const noexcept
        {
            return Color_Space_RGB( r-rhs.r, g-rhs.g, b-rhs.b );
        }
        
    public:
        float   r;
        float   g;
        float   b;
    };
    
    inline
    float
    norm_sqr(
            const Color_Space_RGB& _ ) noexcept
    {
        return sqr(_.r) + sqr(_.g) + sqr(_.b);
    }

}

