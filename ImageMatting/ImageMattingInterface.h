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
	// TODO:  �ڴ�������ķ�����
	static std::shared_ptr<ImageMattingInterface> Create(int deviceID);
	//	srcImage GPU�ڴ�RGB����
	//	dstImage �ڴ�RGB����
	virtual int MattingImage(void* srcImage, int srcWidth, int srcHeight, void* dstImage, int x, int y, int dstWidth, int dstHeight) = 0;
	//	srcImage �ڴ�RGB����
	//	dstImage �ڴ�RGB����
	virtual int MattingImageGpu(void* srcImage, int srcWidth, int srcHeight, void* dstImage, int x, int y, int dstWidth, int dstHeight) = 0;
	// dstImage GPU�Դ��ַ  ��Ҫ����FreeGpuMem�ͷ��Դ�
	// srcImage CPU�ڴ��ַ
	virtual	int MemcpyHostToDev(void** dstImage, int dstLen, void* srcImage, int srcLen)=0;
	// dstImage CPU�ڴ��ַ
	// srcImage GPU�Դ��ַ
	virtual	int MemcpyDevToHost(void* dstImage, int dstLen, void* srcImage, int srcLen)=0;
	// �ͷ��Դ�
	virtual void FreeGpuMem(void* p) = 0;

};
