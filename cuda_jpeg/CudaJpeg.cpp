#include "CudaJpeg.h"
#include "cuda_functions.h"


CCudaJpeg::CCudaJpeg()
{
}


CCudaJpeg::~CCudaJpeg()
{
}

int CCudaJpeg::Init(int gpuID)
{
	m_gpuID = gpuID;
	return 0;
}

int CCudaJpeg::SaveJpeg(const char * filePath, unsigned char * image, int width, int height)
{
	return jpegNPP(filePath, (float*)image, width, height);
}
