#include <cuda.h>
#include <cuda_runtime_api.h>


__device__ float4 convert_one_pixel_to_hsv(uchar4 pixel) {
	float r, g, b, a;
	float h, s, v;
	
	r = pixel.x / 255.0f;
	g = pixel.y / 255.0f;
	b = pixel.z / 255.0f;
	a = pixel.w;
	
	float max = fmax(r, fmax(g, b));
	float min = fmin(r, fmin(g, b));
	float diff = max - min;
	
	v = max;
	
	if(v == 0.0f) { // black
		h = s = 0.0f;
	} else {
		s = diff / v;
		if(diff < 0.001f) { // grey
			h = 0.0f;
		} else { // color
			if(max == r) {
				h = 60.0f * (g - b)/diff;
				if(h < 0.0f) { h += 360.0f; }
			} else if(max == g) {
				h = 60.0f * (2 + (b - r)/diff);
			} else {
				h = 60.0f * (4 + (r - g)/diff);
			}
		}		
	}
	
	return (float4) {h, s, v, a};
}

__global__ void convert_to_hsv(uchar4 *rgb, float4 *hsv, int width, int height) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	if(x < width && y < height) {
		uchar4 rgb_pixel = rgb[x + width*y];
		float4 hsv_pixel = convert_one_pixel_to_hsv(rgb_pixel);
		hsv[x + width*y] = hsv_pixel;
	}
}

extern "C" void convert_to_hsv_wrapper(uchar4 *rgb, float4 *hsv, int width, int height) {
	dim3 threads(128,128);
	dim3 blocks((width + 127)/128, (height + 127)/128);
	
	convert_to_hsv<<<blocks, threads>>>(rgb, hsv, width, height);
}


