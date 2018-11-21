
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

typedef unsigned char	uchar;
typedef unsigned char   uint8;
typedef unsigned int    uint32;
typedef int             int32;
typedef struct uint24
{
	uint8 r;
	uint8 g;
	uint8 b;
};

#define COLOR_COMPONENT_MASK            0x3FF
#define COLOR_COMPONENT_BIT_SIZE        10

#define FIXED_DECIMAL_POINT             24
#define FIXED_POINT_MULTIPLIER          1.0f
#define FIXED_COLOR_COMPONENT_MASK      0xffffffff

typedef enum
{
	ITU601 = 1,
	ITU709 = 2
} eColorSpace;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}


//__global__ void YCrCb2RGBConver(uchar *pYdata, uchar *pUVdata, int stepY, int stepUV, uchar *pImgData, int width, int height, int channels)
__global__ void NV12ToRGB_drvapi(uint32 *srcImage, size_t nSourcePitch, uint32 *dstImage, size_t nDestPitch, uint32 width, uint32 height)
{
	const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	const int tidy = blockIdx.y * blockDim.y + threadIdx.y;
//	const int tidx = blockIdx.x * (blockDim.x << 1) + (threadIdx.x << 1);
//	const int tidy = blockIdx.y *  blockDim.y + threadIdx.y;
	int32 stepY = nSourcePitch;
	int32 stepUV = nSourcePitch;
	uchar* pYdata = (uchar*)srcImage;
	uchar* pUVdata = pYdata + nSourcePitch*height;
	uchar* pImgData = (uchar*)dstImage;
	int channels = 3;

	if (tidx < width && tidy < height)
	{
		int indexY, indexU, indexV;
		uchar Y, U, V;
		indexY = tidy * stepY + tidx;
		Y = pYdata[indexY];

		if (tidx % 2 == 0)
		{
			indexU = tidy / 2 * stepUV + tidx;
			indexV = tidy / 2 * stepUV + tidx + 1;
			U = pUVdata[indexU];
			V = pUVdata[indexV];
		}
		else if (tidx % 2 == 1)
		{
			indexV = tidy / 2 * stepUV + tidx;
			indexU = tidy / 2 * stepUV + tidx - 1;
			U = pUVdata[indexU];
			V = pUVdata[indexV];
		}

		pImgData[(tidy*width + tidx) * channels + 2] = uchar(Y + 1.402 * (V - 128));
		pImgData[(tidy*width + tidx) * channels + 1] = uchar(Y - 0.34413 * (U - 128) - 0.71414*(V - 128));
		pImgData[(tidy*width + tidx) * channels + 0] = uchar(Y + 1.772*(U - 128));
	}
}


__constant__ uint32 constAlpha = ((uint32)0xff << 24);

#define MUL(x,y)    (x*y)
__constant__ float  constHueColorSpaceMat[9] = { 1.1644f, 0.0, 1.5960f, 1.1644f, -0.3918f, -0.8130f, 1.1644f, 2.0172f, 0.0 };


__device__ void YUV2RGB(uint32 *yuvi, float *red, float *green, float *blue)
{
	float luma, chromaCb, chromaCr;

	// Prepare for hue adjustment
	luma = (float)yuvi[0];
	chromaCb = (float)((int32)yuvi[1] - 512.0f);
	chromaCr = (float)((int32)yuvi[2] - 512.0f);

	// Convert YUV To RGB with hue adjustment
	*red = MUL(luma, constHueColorSpaceMat[0]) +
		MUL(chromaCb, constHueColorSpaceMat[1]) +
		MUL(chromaCr, constHueColorSpaceMat[2]);
	*green = MUL(luma, constHueColorSpaceMat[3]) +
		MUL(chromaCb, constHueColorSpaceMat[4]) +
		MUL(chromaCr, constHueColorSpaceMat[5]);
	*blue = MUL(luma, constHueColorSpaceMat[6]) +
		MUL(chromaCb, constHueColorSpaceMat[7]) +
		MUL(chromaCr, constHueColorSpaceMat[8]);
}
__device__ uint32 RGBAPACK_10bit(float red, float green, float blue, uint32 alpha)
{
	uint32 ARGBpixel = 0;

	// Clamp final 10 bit results
	red = min(max(red, 0.0f), 1023.f);
	green = min(max(green, 0.0f), 1023.f);
	blue = min(max(blue, 0.0f), 1023.f);

	// Convert to 8 bit unsigned integers per color component
	ARGBpixel = (((uint32)blue >> 2) |
		(((uint32)green >> 2) << 8) |
		(((uint32)red >> 2) << 16) | (uint32)alpha);

	return  ARGBpixel;
}

__device__ uint32 RGBPACK_10bit(float red, float green, float blue)
{
	uint32 ARGBpixel = 0;

	// Clamp final 10 bit results
	red = min(max(red, 0.0f), 1023.f);
	green = min(max(green, 0.0f), 1023.f);
	blue = min(max(blue, 0.0f), 1023.f);

	// Convert to 8 bit unsigned integers per color component
	ARGBpixel = (((uint32)red >> 2) |
		(((uint32)green >> 2) << 8) |
		(((uint32)blue >> 2) << 16));

	return  ARGBpixel;
}

__device__ uint8 Cloamp_10bit(float data)
{
	data = min(max(data, 0.0f), 1023.f);
	return ((uint32)data >> 2);
//	return (data );
}

__global__ void NV12ToARGB_drvapi(uint32 *srcImage, size_t nSourcePitch,
	uint32 *dstImage, size_t nDestPitch,
	uint32 width, uint32 height)
{
	int32 x, y;
	uint32 yuv101010Pel[2];
	uint32 processingPitch = ((width)+63) & ~63;
	uint32 dstImagePitch = nDestPitch >> 2;
	uint8 *srcImageU8 = (uint8 *)srcImage;

	processingPitch = nSourcePitch;

	// Pad borders with duplicate pixels, and we multiply by 2 because we process 2 pixels per thread
	x = blockIdx.x * (blockDim.x << 1) + (threadIdx.x << 1);
	y = blockIdx.y *  blockDim.y + threadIdx.y;

	if (x >= width)
		return; //x = width - 1;

	if (y >= height)
		return; // y = height - 1;

				// Read 2 Luma components at a time, so we don't waste processing since CbCr are decimated this way.
				// if we move to texture we could read 4 luminance values
	yuv101010Pel[0] = (srcImageU8[y * processingPitch + x]) << 2;
	yuv101010Pel[1] = (srcImageU8[y * processingPitch + x + 1]) << 2;

	uint32 chromaOffset = processingPitch * height;
	int32 y_chroma = y >> 1;

	if (y & 1)  // odd scanline ?
	{
		uint32 chromaCb;
		uint32 chromaCr;

		chromaCb = srcImageU8[chromaOffset + y_chroma * processingPitch + x];
		chromaCr = srcImageU8[chromaOffset + y_chroma * processingPitch + x + 1];

		if (y_chroma < ((height >> 1) - 1)) // interpolate chroma vertically
		{
			chromaCb = (chromaCb + srcImageU8[chromaOffset + (y_chroma + 1) * processingPitch + x] + 1) >> 1;
			chromaCr = (chromaCr + srcImageU8[chromaOffset + (y_chroma + 1) * processingPitch + x + 1] + 1) >> 1;
		}

		yuv101010Pel[0] |= (chromaCb << (COLOR_COMPONENT_BIT_SIZE + 2));
		yuv101010Pel[0] |= (chromaCr << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));

		yuv101010Pel[1] |= (chromaCb << (COLOR_COMPONENT_BIT_SIZE + 2));
		yuv101010Pel[1] |= (chromaCr << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));
	}
	else
	{
		yuv101010Pel[0] |= ((uint32)srcImageU8[chromaOffset + y_chroma * processingPitch + x] << (COLOR_COMPONENT_BIT_SIZE + 2));
		yuv101010Pel[0] |= ((uint32)srcImageU8[chromaOffset + y_chroma * processingPitch + x + 1] << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));

		yuv101010Pel[1] |= ((uint32)srcImageU8[chromaOffset + y_chroma * processingPitch + x] << (COLOR_COMPONENT_BIT_SIZE + 2));
		yuv101010Pel[1] |= ((uint32)srcImageU8[chromaOffset + y_chroma * processingPitch + x + 1] << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));
	}

	// this steps performs the color conversion
	uint32 yuvi[6];
	float red[2], green[2], blue[2];

	yuvi[0] = (yuv101010Pel[0] & COLOR_COMPONENT_MASK);
	yuvi[1] = ((yuv101010Pel[0] >> COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK);
	yuvi[2] = ((yuv101010Pel[0] >> (COLOR_COMPONENT_BIT_SIZE << 1)) & COLOR_COMPONENT_MASK);

	yuvi[3] = (yuv101010Pel[1] & COLOR_COMPONENT_MASK);
	yuvi[4] = ((yuv101010Pel[1] >> COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK);
	yuvi[5] = ((yuv101010Pel[1] >> (COLOR_COMPONENT_BIT_SIZE << 1)) & COLOR_COMPONENT_MASK);

	// YUV to RGB Transformation conversion
	YUV2RGB(&yuvi[0], &red[0], &green[0], &blue[0]);
	YUV2RGB(&yuvi[3], &red[1], &green[1], &blue[1]);
//	printf("cuda out x,y=[%d,%d], dstImagePitch=%d, processingPitch=%d\n", x, y, dstImagePitch, processingPitch);


	// Clamp the results to RGBA
	dstImage[y * dstImagePitch + x] = RGBPACK_10bit(red[0], green[0], blue[0]);
	dstImage[y * dstImagePitch + x + 1] = RGBPACK_10bit(red[1], green[1], blue[1]);
}

__global__ void NV12ToRGB_drvapi2(uint32 *srcImage, size_t nSourcePitch,
	uint32 *dstImage, size_t nDestPitch,
	uint32 width, uint32 height)
{
	int32 x, y;
	uint32 yuv101010Pel[2];
	uint32 processingPitch = ((width)+63) & ~63;
	uint32 dstImagePitch = nDestPitch / 3;
	uint8 *srcImageU8 = (uint8 *)srcImage;
	uint24 *dstImageU8 = (uint24 *)dstImage;

	processingPitch = nSourcePitch;

	// Pad borders with duplicate pixels, and we multiply by 2 because we process 2 pixels per thread
	x = blockIdx.x * (blockDim.x << 1) + (threadIdx.x << 1);
//	x = blockIdx.x * (blockDim.x) + (threadIdx.x);
	y = blockIdx.y *  blockDim.y + threadIdx.y;

	if (x >= width)
		return; //x = width - 1;

	if (y >= height)
		return; // y = height - 1;

				// Read 2 Luma components at a time, so we don't waste processing since CbCr are decimated this way.
				// if we move to texture we could read 4 luminance values
	yuv101010Pel[0] = (srcImageU8[y * processingPitch + x]) << 2;
	yuv101010Pel[1] = (srcImageU8[y * processingPitch + x + 1]) << 2;

	uint32 chromaOffset = processingPitch * height;
	int32 y_chroma = y >> 1;

	if (y & 1)  // odd scanline ?
	{
		uint32 chromaCb;
		uint32 chromaCr;

		chromaCb = srcImageU8[chromaOffset + y_chroma * processingPitch + x];
		chromaCr = srcImageU8[chromaOffset + y_chroma * processingPitch + x + 1];

		if (y_chroma < ((height >> 1) - 1)) // interpolate chroma vertically
		{
			chromaCb = (chromaCb + srcImageU8[chromaOffset + (y_chroma + 1) * processingPitch + x] + 1) >> 1;
			chromaCr = (chromaCr + srcImageU8[chromaOffset + (y_chroma + 1) * processingPitch + x + 1] + 1) >> 1;
		}

		yuv101010Pel[0] |= (chromaCb << (COLOR_COMPONENT_BIT_SIZE + 2));
		yuv101010Pel[0] |= (chromaCr << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));

		yuv101010Pel[1] |= (chromaCb << (COLOR_COMPONENT_BIT_SIZE + 2));
		yuv101010Pel[1] |= (chromaCr << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));
	}
	else
	{
		yuv101010Pel[0] |= ((uint32)srcImageU8[chromaOffset + y_chroma * processingPitch + x] << (COLOR_COMPONENT_BIT_SIZE + 2));
		yuv101010Pel[0] |= ((uint32)srcImageU8[chromaOffset + y_chroma * processingPitch + x + 1] << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));

		yuv101010Pel[1] |= ((uint32)srcImageU8[chromaOffset + y_chroma * processingPitch + x] << (COLOR_COMPONENT_BIT_SIZE + 2));
		yuv101010Pel[1] |= ((uint32)srcImageU8[chromaOffset + y_chroma * processingPitch + x + 1] << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));
	}

	// this steps performs the color conversion
	uint32 yuvi[6];
	float red[2], green[2], blue[2];

	yuvi[0] = (yuv101010Pel[0] & COLOR_COMPONENT_MASK);
	yuvi[1] = ((yuv101010Pel[0] >> COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK);
	yuvi[2] = ((yuv101010Pel[0] >> (COLOR_COMPONENT_BIT_SIZE << 1)) & COLOR_COMPONENT_MASK);

	yuvi[3] = (yuv101010Pel[1] & COLOR_COMPONENT_MASK);
	yuvi[4] = ((yuv101010Pel[1] >> COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK);
	yuvi[5] = ((yuv101010Pel[1] >> (COLOR_COMPONENT_BIT_SIZE << 1)) & COLOR_COMPONENT_MASK);

	// YUV to RGB Transformation conversion
	YUV2RGB(&yuvi[0], &red[0], &green[0], &blue[0]);
	YUV2RGB(&yuvi[3], &red[1], &green[1], &blue[1]);

	dstImageU8[y * dstImagePitch + x].r = Cloamp_10bit(red[0]);
	dstImageU8[y * dstImagePitch + x].g = Cloamp_10bit(green[0]);
	dstImageU8[y * dstImagePitch + x].b = Cloamp_10bit(blue[0]);
	dstImageU8[y * dstImagePitch + x + 1].r = Cloamp_10bit(red[1]);
	dstImageU8[y * dstImagePitch + x + 1].g = Cloamp_10bit(green[1]);
	dstImageU8[y * dstImagePitch + x + 1].b = Cloamp_10bit(blue[1]);
	//	dstImageU8[(y+1) * dstImagePitch + x] = Cloamp_10bit(green[0]);
//	dstImageU8[y * dstImagePitch + x + 2] = Cloamp_10bit(blue[0]);
//	dstImageU8[y * dstImagePitch + x + 3] = Cloamp_10bit(red[1]);
//	dstImageU8[y * dstImagePitch + x + 4] = Cloamp_10bit(green[1]);
//	dstImageU8[y * dstImagePitch + x + 5] = Cloamp_10bit(blue[1]);
//	printf("x,y=[%d, %d],dstImagePitch = %d, pos = %d\n", x, y, dstImagePitch, y * dstImagePitch + x);
}

int main()
{
	cudaError_t cudaStatus;
	int width = 1920;
	int height = 1080;
	int size = 0;
	FILE* pf = fopen("d:\\test.nv12", "rb");
	unsigned char* pData = NULL;
	if (pf)
	{
		fseek(pf, 0L, SEEK_END);
		size = ftell(pf);
		fseek(pf, 0L, SEEK_SET);
		pData = new unsigned char[size];
		fread(pData, size, 1, pf);
		fclose(pf);
	}
	if (pData)
	{
		int imageSize = width*height * 4;
		unsigned char* pnv12 = NULL;
		unsigned char* pImage = NULL;
		unsigned char* pImage2 = new unsigned char[imageSize];
		cudaStatus = cudaMalloc((void**)&pnv12, size);
		cudaStatus = cudaMalloc((void**)&pImage, imageSize);
		cudaStatus = cudaMemcpy(pnv12, pData, size, cudaMemcpyHostToDevice);
		dim3 block(32, 16, 1);
//		dim3 grid(width, height, 1);
		dim3 grid((width + (2 * block.x - 1)) / (2 * block.x), (height + (block.y - 1)) / block.y, 1);
//		dim3 grid((width + (block.x - 1)) / (block.x), (height + (block.y - 1)) / block.y, 1);
		//		NV12ToRGB_drvapi <<<grid, block, 0>>> ((uint32*)pnv12, 2048, (uint32*)pImage, width * 4, width, height);
//		NV12ToARGB_drvapi <<<grid, block, 0 >>> ((uint32*)pnv12, 2048, (uint32*)pImage, width * 4, width, height);
		NV12ToRGB_drvapi2 << <grid, block, 0 >> > ((uint32*)pnv12, 2048, (uint32*)pImage, width * 3, width, height);
		cudaStatus = cudaMemcpy(pImage2, pImage, width*height * 4, cudaMemcpyDeviceToHost);
		FILE* pf2 = fopen("d:\\image.rgba", "wb");
		if (pf2)
		{
			fwrite(pImage2, width*height * 4, 1, pf2);
			fclose(pf2);
		}
		delete[] pImage2;
		cudaFree(pnv12);
		cudaFree(pImage);
	}
	delete[] pData;
	return 0;
}


//int main()
//{
//    const int arraySize = 5;
//    const int a[arraySize] = { 1, 2, 3, 4, 5 };
//    const int b[arraySize] = { 10, 20, 30, 40, 50 };
//    int c[arraySize] = { 0 };
//
//    // Add vectors in parallel.
//    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addWithCuda failed!");
//        return 1;
//    }
//
//    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//        c[0], c[1], c[2], c[3], c[4]);
//
//    // cudaDeviceReset must be called before exiting in order for profiling and
//    // tracing tools such as Nsight and Visual Profiler to show complete traces.
//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }
//
//    return 0;
//}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
