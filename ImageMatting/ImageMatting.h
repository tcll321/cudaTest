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
	// TODO:  �ڴ�������ķ�����
	int Init(int deviceID);
	//	srcImage GPU�ڴ�RGB����
	//	dstImage �ڴ�RGB����
	virtual int MattingImage(void* srcImage, int srcWidth, int srcHeight, void* dstImage, int x, int y, int dstWidth, int dstHeight);

	//	srcImage �ڴ�RGB����
	//	dstImage �ڴ�RGB����
	virtual int MattingImageGpu(void* srcImage, int srcWidth, int srcHeight, void* dstImage, int x, int y, int dstWidth, int dstHeight);

	// dstImage GPU�Դ��ַ
	// srcImage CPU�ڴ��ַ
	virtual	int MemcpyHostToDev(void** dstImage, int dstLen, void* srcImage, int srcLen);
	// dstImage CPU�ڴ��ַ
	// srcImage GPU�Դ��ַ
	virtual	int MemcpyDevToHost(void* dstImage, int dstLen, void* srcImage, int srcLen);
	// �ͷ��Դ�
	virtual void FreeGpuMem(void* p);

private:
// 	CUcontext     m_oContext;
// 	CUdevice      m_oDevice;
	int			  m_deviceID;
};
