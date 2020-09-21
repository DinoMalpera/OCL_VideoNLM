# OCL VideoNLM

Implementation of Non-Local Means noise reduction algorithm for video in OpenCL. Where the basic NLM algorithm collects statistics from a single image (single frame - the one being denoised), video NLM takes advantage of the fact that successive frames in video are highly correlated.<br/>
Expanded on my [SISD implementation](https://github.com/DinoMalpera/videoNLM).<br/>
<p align="center">
<img src="images/vnlm.jpg">
</p>

### Dependencies

* C++20 ready compiler<br/>
* Boost
* OpenCL

### Status

Currently OCL_VNLM is in it's first iteration; basic functionality, unoptimized.

### Example

Example use of VNLM for single image using TIFFIO library.

```c++
using namespace VNLM;

std::transform(
	raster,
	raster+number_of_pixels,
	frame_data_holder.begin(),
	[](const unsigned int& c) -> auto
	{
		return Color_Space_RGB(
			TIFFGetR(c)/255.0f,
			TIFFGetG(c)/255.0f,
			TIFFGetB(c)/255.0f
		);
	}
	);

Frame<Color_Space_RGB> fr( &frame_data_holder[0], frameSize, 0);
Frame<Color_Space_RGB> outputPic( &output_frame[0], frameSize, 0 );

FrameSequence<Color_Space_RGB> frameSequence;
frameSequence.push_back( fr );

NLMdenoiser nlm;
nlm.Denoise( frameSequence, outputPic );
```


