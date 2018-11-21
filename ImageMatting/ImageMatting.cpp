#include "ImageMatting.h"
#include <windows.h>
#include <stdio.h>
#include "def.h"

extern "C" cudaError_t kernel_MattingPicture(void * srcImage, int srcWidth, int srcHeight, void * dstImage, int x, int y, int dstWidth, int dstHeight);

ImageMatting::ImageMatting()
	:m_deviceID(0)
{

}
ImageMatting::~ImageMatting()
{
// 	cuCtxDestroy(m_oContext);
}

int ImageMatting::Init(int deviceID)
{
	cudaError_t cudaStatus;
	CUresult status;

	// Choose which GPU to run on, change this on a multi-GPU system.
// 	cudaStatus = cudaSetDevice(deviceID);
// 	if (cudaStatus != cudaSuccess) {
// 		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
// 		return -1;
// 	}
// 	cudaStatus = cuDeviceGet(&m_oDevice, deviceID);
// 		if (cudaStatus != cudaSuccess) {
// 			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
// 			return -1;
// 		}
// 	status = cuCtxCreate(&m_oContext, CU_CTX_BLOCKING_SYNC, deviceID);
	m_deviceID = deviceID;
	return 0;
}

int ImageMatting::MattingImage(void * srcImage, int srcWidth, int srcHeight, void * dstImage, int x, int y, int dstWidth, int dstHeight)
{
	cudaError_t cudaStatus;
	unsigned char* pDstImage = NULL;
	int	nDevId;
	int dstImageSize = dstWidth*dstHeight * 4;
	cudaStatus = cudaSetDevice(m_deviceID);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaGetDevice failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}

	cudaStatus = cudaMalloc((void**)&pDstImage, dstImageSize);
	if (cudaStatus != cudaSuccess)
	{
		cudaFree(pDstImage);
		fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}
	cudaStatus = kernel_MattingPicture(srcImage, srcWidth, srcHeight, pDstImage, x, y, dstWidth, dstHeight);
	if (cudaStatus != cudaSuccess)
	{
		cudaFree(pDstImage);
		fprintf(stderr, "kernel_MattingPicture failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}
	cudaStatus = cudaMemcpy(dstImage, pDstImage, dstWidth*dstHeight * 3, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		cudaFree(pDstImage);
		fprintf(stderr, "memcpy device to host failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		cudaFree(pDstImage);
		fprintf(stderr, "memcpy host to device failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}
	cudaFree(pDstImage);
	return 0;
}

int ImageMatting::MattingImageGpu(void * srcImage, int srcWidth, int srcHeight, void * dstImage, int x, int y, int dstWidth, int dstHeight)
{
	cudaError_t cudaStatus;
	unsigned char* pDstImage = NULL;
	unsigned char* pcuSrcImage = NULL;
	int	nDevId;
	int dstImageSize = dstWidth*dstHeight * 4;
	cudaStatus = cudaSetDevice(m_deviceID);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaGetDevice failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}

	//CUresult result = cuCtxPushCurrent(m_oContext);
	cudaStatus = cudaMalloc((void**)&pDstImage, dstImageSize);
	cudaStatus = cudaMalloc((void**)&pcuSrcImage, dstImageSize);
	cudaStatus = cudaMemcpy(pcuSrcImage, srcImage, dstWidth*dstHeight * 3, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cudaFree(pcuSrcImage);
		cudaFree(pDstImage);
		fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}
	cudaStatus = kernel_MattingPicture(pcuSrcImage, srcWidth, srcHeight, pDstImage, x, y, dstWidth, dstHeight);
	if (cudaStatus != cudaSuccess)
	{
		cudaFree(pDstImage);
		fprintf(stderr, "kernel_MattingPicture failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}
	cudaStatus = cudaMemcpy(dstImage, pDstImage, dstWidth*dstHeight * 3, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		cudaFree(pcuSrcImage);
		cudaFree(pDstImage);
		fprintf(stderr, "memcpy device to host failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		cudaFree(pcuSrcImage);
		cudaFree(pDstImage);
		fprintf(stderr, "memcpy host to device failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}
	cudaFree(pcuSrcImage);
	cudaFree(pDstImage);
	return 0;
}

int ImageMatting::MemcpyHostToDev(void ** dstImage, int dstLen, void * srcImage, int srcLen)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(m_deviceID);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaGetDevice failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}
	cudaStatus = cudaMalloc(dstImage, dstLen);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}
	cudaStatus = cudaMemcpy(*dstImage, srcImage, srcLen, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cudaFree(dstImage);
		fprintf(stderr, "memcpy host to device failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		cudaFree(dstImage);
		fprintf(stderr, "memcpy host to device failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}
	return 0;
}

int ImageMatting::MemcpyDevToHost(void * dstImage, int dstLen, void * srcImage, int srcLen)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(m_deviceID);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaGetDevice failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}

	cudaStatus = cudaMemcpy(dstImage, srcImage, srcLen, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "memcpy host to device failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "memcpy host to device failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}
	return 0;
}

void ImageMatting::FreeGpuMem(void * p)
{
	if (p)
	{
		cudaFree(p);
		p = NULL;
	}
}

