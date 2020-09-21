#pragma once

#include <concepts>

namespace VNLM
{

    /*  Computable Color concept
    */
    template <typename T>
    concept ComputableColor = requires(T c)
    {
        T();
        T(2.0);
        { c*2.0 } -> std::convertible_to<T>;
        { c+=c  } -> std::same_as<T&>;
        { c/2.0 } -> std::convertible_to<T>;
        { c-c   } -> std::same_as<T>;
    };

}

