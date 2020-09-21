#pragma once

namespace VNLM
{

    /*  Aggregate class; holds pixel coordinates.
     * */
    struct Pixel_Coord
    {
        unsigned int x;
        unsigned int y;
    };
    
    /*  Needed by Every_Pixel_in_SearchWindow_of_Every_Frame::iterator to compute
     *  offsets to search window 
     */
    inline
    Pixel_Coord
    operator+(
            const Pixel_Coord& lhs,
            const Pixel_Coord& rhs ) noexcept
    {
        return {    lhs.x + rhs.x,
                    lhs.y + rhs.y };
    }

}
