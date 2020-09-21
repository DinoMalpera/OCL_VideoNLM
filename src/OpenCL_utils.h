#pragma once

#include <string>

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
//#define __CL_ENABLE_EXCEPTIONS
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>

cl::Device
get_first_available_device();

std::string
load_kernel_source(
        const std::string& file_name );
        

