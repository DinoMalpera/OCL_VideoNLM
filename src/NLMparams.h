#pragma once

namespace VNLM
{

    /*  Input parameters of the NLM algorithm.
     */
    struct NLMparams
    {
        unsigned int    patch_radius;
        unsigned int    search_window_radius;
        float           standard_deviation_of_noise;
        float           filtering_parameter;
    };

}

