#include <iostream>
#include <fstream>
#include <string>
#include "OpenCL_utils.h"

namespace {

std::vector<cl::Device>
find_devices(
        const std::vector<cl::Platform>& platforms )
{
    constexpr cl_device_type device_type = CL_DEVICE_TYPE_GPU;

    std::vector<cl::Device> devices;
    for( const auto& platform : platforms )
    {
        std::vector<cl::Device> platform_devices;
        
        try
        {
            platform.getDevices( device_type, &platform_devices );
            
            for( const auto& device : platform_devices )
            {
                if ( !device.getInfo<CL_DEVICE_AVAILABLE>() )
                {
                    continue;
                }
                
                devices.push_back( device );
            }
        }
        catch(...)
        {
        }
    }
    
    return devices;
}

void
print_devices(
        const std::vector<cl::Device>& devices )
{
    std::cout << "Devices found:\n";
    for( const auto it : devices )
    {
        std::cout << "- " << it.getInfo<CL_DEVICE_NAME>() << "\n";
    }
    std::cout << std::endl;
}

}
    
cl::Device
get_first_available_device()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get( &platforms );
    
    if ( platforms.empty() )
    {
        std::cerr << "No OpenCL platforms found!" << std::endl;
        throw 1;
    }
    
    std::vector<cl::Device> devices( find_devices( platforms ) );
    
    if ( devices.empty() )
    {
        std::cerr << "No OpenCL devices (of required type) found!" << std::endl;
        throw 2;
    }
    
    print_devices( devices );
    
    return std::move(devices[0u]);
}

std::string
load_kernel_source(
        const std::string& file_name )
{
    std::ifstream ifs( file_name, std::ifstream::in );
    
    std::string source_code;
    std::string source_line;
    
    while( std::getline( ifs, source_line ) )
    {
        source_code += source_line;
        source_code.push_back( '\n' );
    }
    
    return source_code;
}

