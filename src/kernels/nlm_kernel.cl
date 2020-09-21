
float
color_distance(
        const float3 c,
        const float3 a )
{
    const float3 d = (c - a);
    return dot( d, d );
}

float
gauss(
        const float x,
        const float std_dev,
        const float filtering_param )
{
    return
        exp(
            - max( x - 2*std_dev, 0.0f)
            /
            filtering_param );
}

float
compute_patch_weight(
        __global const float3* fr_c,
        __global const float3* fr_a,
        const int2             patch_center_c,
        const int2             patch_center_a,
        const int              patch_radius,
        const int2             frame_sizeW,
        const float            std_dev,
        const float            filtering_param )
{
    float       acc              = 0.0f;
    const int   patch_diameter   = 2*patch_radius + 1;
    const int2  start_c          = patch_center_c - patch_radius;
    const int2  start_a          = patch_center_a - patch_radius;
    const float patch_total_size = patch_diameter * patch_diameter;
    
    for( int y=0; y<patch_diameter; ++y )
    {
        int posc = mad24( y+start_c.y, frame_sizeW.x, start_c.x );
        int posa = mad24( y+start_a.y, frame_sizeW.x, start_a.x );
        const int posc_end = posc + patch_diameter;
        for( ; posc<posc_end; ++posc, ++posa )
        {
            acc += color_distance(
                fr_c[posc],
                fr_a[posa] );
        }
    }
    
    acc /= patch_total_size;
    
    const float weight =
        gauss(
            acc,
            std_dev,
            filtering_param );
    
    return weight;
}

__kernel
void
compute_search_window(
        __global const float3* fr_c,
        __global const float3* fr_a,
        __global       float4* fr_output,
        const int              search_radius,
        const int              patch_radius,
        const int2             frame_size,
        const float            std_dev,
        const float            filtering_param )
{
    const int  gidx           = get_global_id(0);
    const int  gidy           = get_global_id(1);
    
    const int2 frame_sizeE    = frame_size + patch_radius;
    const int2 frame_sizeW    = frame_size + 2*patch_radius;
    
    const int2 patch_center_c = (int2)( gidx, gidy) + patch_radius;

    const int2 start          = max( patch_center_c - (int2)search_radius          , (int2)patch_radius );
    const int2 end            = min( patch_center_c + (int2)search_radius + (int2)1, (int2)frame_sizeE );
    float4     acc            = 0.0f;
    
    for( int y=start.y; y<end.y; ++y )
    for( int x=start.x; x<end.x; ++x )
    {
        const int2 patch_center_a = (int2)(x,y);
        
        const float weight =
            compute_patch_weight(
                fr_c,
                fr_a,
                patch_center_c,
                patch_center_a,
                patch_radius,
                frame_sizeW,
                std_dev,
                filtering_param );
                
        const int posa = mad24( patch_center_a.y, frame_sizeW.x, patch_center_a.x );
        
        acc += (float4)( fr_a[posa]*weight, weight );
    }
    
    const int pos_o = mad24( gidy, frame_size.x, gidx );
        
    fr_output[pos_o] += acc;
}

