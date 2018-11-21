#pragma once
#include <memory>

#ifdef CUDA_JPEG_EXPORTS
#define  CUDA_JPEG_API __declspec(dllexport)
#else
#define  CUDA_JPEG_API __declspec(dllimport)
#endif


class CUDA_JPEG_API CCudaJpegInterface
{
public:
	CCudaJpegInterface();
	static std::shared_ptr<CCudaJpegInterface> Create(int deviceID);

	virtual	int SaveJpeg(const char* filePath, unsigned char* image, int width, int height) = 0;
};

