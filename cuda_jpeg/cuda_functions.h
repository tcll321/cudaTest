
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include <string.h>
#include <math.h>

#include <cuda.h>

namespace cuda_common
{
	cudaError_t RGB2YUV(float* d_srcRGB, int src_width, int src_height, 
						unsigned char* Y, size_t yPitch, int yWidth, int yHeight,
						unsigned char* U, size_t uPitch, int uWidth, int uHeight,
						unsigned char* V, size_t vPitch, int vWidth, int vHeight);
}

// d_srcRGB数据排列形式为  BBBBBB......GGGGGGG......RRRRRRRR......
int jpegNPP(const char *szOutputFile, float* d_srcRGB, int img_width, int img_height);

