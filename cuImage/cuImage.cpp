// cuImage.cpp : ���� DLL Ӧ�ó���ĵ���������
//

#include "stdafx.h"
#include "cuImage.h"


// ���ǵ���������һ��ʾ����
CUIMAGE_API int fncuImage(void)
{
    return 42;
}

CUIMAGE_API int MattingImage(void * srcImage, int srcWidth, int srcHeight, BYTE * dstImage, int x, int y, int dstWidth, int dstHeight)
{
	if (srcImage == NULL)
		return -1;

	return 0;
}

int CcuImage::Init(int deviceID)
{
	return 0;
}
