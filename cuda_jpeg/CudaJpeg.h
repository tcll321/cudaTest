#pragma once
#include "CudaJpegInterface.h"
class CCudaJpeg :
	public CCudaJpegInterface
{
public:
	CCudaJpeg();
	~CCudaJpeg();

	int Init(int gpuID);

	virtual	int SaveJpeg(const char* filePath, unsigned char* image, int width, int height);

private:
	int m_gpuID;
};

