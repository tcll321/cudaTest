#include "ImageMattingInterface.h"
#include "ImageMatting.h"

ImageMattingInterface::ImageMattingInterface()
{

}

std::shared_ptr<ImageMattingInterface> ImageMattingInterface::Create(int deviceID)
{
	ImageMatting* pImageMat = new ImageMatting();
	if (pImageMat->Init(deviceID) != 0)
	{
		delete pImageMat;
		return NULL;
	}
	std::shared_ptr<ImageMattingInterface> p(pImageMat);
	return p;
}