#include "CudaJpegInterface.h"
#include "CudaJpeg.h"


CCudaJpegInterface::CCudaJpegInterface()
{
}

std::shared_ptr<CCudaJpegInterface> CCudaJpegInterface::Create(int deviceID)
{
	CCudaJpeg* pCudaJpeg = new CCudaJpeg();
	if (pCudaJpeg->Init(deviceID) != 0)
	{
		delete pCudaJpeg;
		return NULL;
	}
	std::shared_ptr<CCudaJpegInterface> p(pCudaJpeg);
	return p;
}
