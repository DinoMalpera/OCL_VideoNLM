#pragma once

/*  Common helpers 
 * */

namespace VNLM
{

    /*  Square of a squareable type.
     * */
    template <typename T>
    constexpr
    T
    sqr(
            const T _ ) noexcept
    {
        return _ * _;
    }

}
