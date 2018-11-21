#pragma once

#include "ImageMattingInterface.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class ImageMatting :
	public ImageMattingInterface
{
public:
	ImageMatting(void);
	~ImageMatting();
	// TODO:  在此添加您的方法。
	int Init(int deviceID);
	//	srcImage GPU内存RGB数据
	//	dstImage 内存RGB数据
	virtual int MattingImage(void* srcImage, int srcWidth, int srcHeight, void* dstImage, int x, int y, int dstWidth, int dstHeight);

	//	srcImage 内存RGB数据
	//	dstImage 内存RGB数据
	virtual int MattingImageGpu(void* srcImage, int srcWidth, int srcHeight, void* dstImage, int x, int y, int dstWidth, int dstHeight);

	// dstImage GPU显存地址
	// srcImage CPU内存地址
	virtual	int MemcpyHostToDev(void** dstImage, int dstLen, void* srcImage, int srcLen);
	// dstImage CPU内存地址
	// srcImage GPU显存地址
	virtual	int MemcpyDevToHost(void* dstImage, int dstLen, void* srcImage, int srcLen);
	// 释放显存
	virtual void FreeGpuMem(void* p);

private:
// 	CUcontext     m_oContext;
// 	CUdevice      m_oDevice;
	int			  m_deviceID;
};
