#pragma once
#include <memory>

#ifdef IMAGEMATTING_EXPORTS
#define IMAGEMATTING_API __declspec(dllexport)
#else
#define IMAGEMATTING_API __declspec(dllimport)
#endif


class IMAGEMATTING_API ImageMattingInterface {
public:
	ImageMattingInterface(void);
	// TODO:  在此添加您的方法。
	static std::shared_ptr<ImageMattingInterface> Create(int deviceID);
	//	srcImage GPU内存RGB数据
	//	dstImage 内存RGB数据
	virtual int MattingImage(void* srcImage, int srcWidth, int srcHeight, void* dstImage, int x, int y, int dstWidth, int dstHeight) = 0;
	//	srcImage 内存RGB数据
	//	dstImage 内存RGB数据
	virtual int MattingImageGpu(void* srcImage, int srcWidth, int srcHeight, void* dstImage, int x, int y, int dstWidth, int dstHeight) = 0;
	// dstImage GPU显存地址  需要调用FreeGpuMem释放显存
	// srcImage CPU内存地址
	virtual	int MemcpyHostToDev(void** dstImage, int dstLen, void* srcImage, int srcLen)=0;
	// dstImage CPU内存地址
	// srcImage GPU显存地址
	virtual	int MemcpyDevToHost(void* dstImage, int dstLen, void* srcImage, int srcLen)=0;
	// 释放显存
	virtual void FreeGpuMem(void* p) = 0;

};
