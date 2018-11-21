
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

typedef unsigned int    uint32;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__device__ uint32 Cloamp_10bit(float data)
{
	return ((uint32)data >> 2);
}

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
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

typedef unsigned int    uint32;
typedef int             int32;
typedef unsigned char   uint8;
typedef struct uint24
{
	uint8 r;
	uint8 g;
	uint8 b;
};

__global__ void cutPicture(uint32* dstImage, uint32* srcImage, int srcWidth, int srcHeight, int cx, int cy, int width)
{
	int32 x, y;
	int srcPitch = srcWidth * 3;
	int dstPitch = width * 3;
	uint24 *srcImageU8 = (uint24 *)srcImage;
	uint24 *dstImageU8 = (uint24 *)dstImage;

	x = blockIdx.x * (blockDim.x << 1) + (threadIdx.x << 1);
	y = blockIdx.y *  blockDim.y + threadIdx.y;

	if ((x) >= width)
		return; //x = width - 1;

	if ((y) >= width)
		return; // y = height - 1;

	dstImageU8[y * width + x].r = srcImageU8[(y+cy) * srcWidth + x + cx].r;
	dstImageU8[y * width + x].g = srcImageU8[(y+cy) * srcWidth + x + cx].g;
	dstImageU8[y * width + x].b = srcImageU8[(y+cy) * srcWidth + x + cx].b;
	dstImageU8[y * width + x + 1].r = srcImageU8[(y + cy) * srcWidth + x + cx + 1].r;
	dstImageU8[y * width + x + 1].g = srcImageU8[(y + cy) * srcWidth + x + cx + 1].g;
	dstImageU8[y * width + x + 1].b = srcImageU8[(y + cy) * srcWidth + x + cx + 1].b;
}

int main()
{
	cudaError_t cudaStatus;
	int size = 0;
	int x = 900;
	int y = 400;
	int width = 400;
	int dstImageSize = width*width * 4;
	FILE* pf = fopen("d:\\image.rgba", "rb");
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
		unsigned char* pSrcImage = NULL;
		unsigned char* pDstImage = NULL;
		unsigned char* pHImage = new unsigned char[dstImageSize];
		cudaStatus = cudaMalloc((void**)&pSrcImage, size);
		cudaStatus = cudaMalloc((void**)&pDstImage, dstImageSize);
		cudaStatus = cudaMemcpy(pSrcImage, pData, size, cudaMemcpyHostToDevice);
		dim3 block(32, 16, 1);
		dim3 grid((width + (2 * block.x - 1)) / (2 * block.x), (width + (block.y - 1)) / block.y, 1);
		cutPicture << <grid, block, 0 >> > ((uint32*)pDstImage, (uint32*)pSrcImage, 1920, 1080, x, y, width);
		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		}
		cudaStatus = cudaMemcpy(pHImage, pDstImage, dstImageSize, cudaMemcpyDeviceToHost);
		FILE* pf2 = fopen("d:\\cutimage.rgb", "wb");
		if (pf2)
		{
			fwrite(pHImage, dstImageSize, 1, pf2);
			fclose(pf2);
		}
		delete[] pHImage;
		cudaFree(pSrcImage);
		cudaFree(pDstImage);
		delete[] pData;
	}
	return 0;
}