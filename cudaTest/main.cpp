#include <windows.h>
#include <stdio.h>
#include <emmintrin.h>
#include "register.h"

typedef unsigned int    uint32;
typedef unsigned char   uint8;


void CoverbySSE2(int width, int height, int pitch, unsigned char* pData, unsigned char* image)
{
	__m128i c0 = _mm_setzero_si128();
	__m128i c128 = _mm_set1_epi16(128);
	__m128i c128_32 = _mm_set1_epi32(128);
	__m128i c16 = _mm_set1_epi16(16);
	__m128i c255 = _mm_set1_epi16(255);
	__m128i c_1_1596 = _mm_set1_epi32(0x199012a);
	__m128i c_1_2017 = _mm_set1_epi32(0x204012a);
	__m128i c_0_392 = _mm_set1_epi32(0xff9c0000);
	__m128i c_1_813 = _mm_set1_epi32(0xff30012a);

	for (int y = 0; y < height; y++)
	{
		BYTE* dest = (BYTE*)image + width * y;
		BYTE* srcY = pData + pitch * y;
		BYTE* srcUV = (pData + width*height) + pitch * (y / 2);
		for (int x = 0; x < width; x += 4)
		{
			//Y0Y1Y2Y30000 - 16
			__m128i Ymm = _mm_sub_epi16(_mm_unpacklo_epi8(_mm_cvtsi32_si128(*(int*)(srcY + x)), c0), c16);
			//U0V0U2V20000 - 128
			__m128i UVmm = _mm_sub_epi16(_mm_unpacklo_epi8(_mm_cvtsi32_si128(*(int*)(srcUV + x)), c0), c128);
			//U0U0U2U20000
			__m128i Umm = _mm_shufflelo_epi16(UVmm, _MM_SHUFFLE(2, 2, 0, 0));
			//V0V0V2V20000
			__m128i Vmm = _mm_shufflelo_epi16(UVmm, _MM_SHUFFLE(3, 3, 1, 1));
			//Y0V0Y1V0Y2V2Y3V2
			__m128i YVmm = _mm_unpacklo_epi16(Ymm, Vmm);
			//Y0U0Y1U0Y2U2Y3U2
			__m128i YUmm = _mm_unpacklo_epi16(Ymm, Umm);

			__m128i Rmm = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(YVmm, c_1_1596), c128_32), 8);
			__m128i Bmm = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(YUmm, c_1_2017), c128_32), 8);
			__m128i Gmm = _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(_mm_madd_epi16(YVmm, c_1_813), _mm_madd_epi16(YUmm, c_0_392)), c128_32), 8);
			Rmm = _mm_slli_epi32(_mm_and_si128(Rmm, _mm_cmpgt_epi32(Rmm, c0)), 16);
			Bmm = _mm_and_si128(Bmm, _mm_cmpgt_epi32(Bmm, c0));
			Gmm = _mm_slli_epi32(_mm_min_epi16(_mm_and_si128(Gmm, _mm_cmpgt_epi32(Gmm, c0)), c255), 8);
			*(__m128i*)dest = _mm_or_si128(_mm_min_epi16(_mm_or_si128(Rmm, Bmm), c255), Gmm);
			dest += 16;
		}
	}
}

int main(int argc, char **argv)
{

	int width = 1920;
	int height = 1080;
	int size = 0;
	FILE* pf = fopen("d:\\test.nv12", "rb");
	unsigned char* pData = NULL;
	if (pf)
	{
		fseek(pf, 0L, SEEK_END);
		size = ftell(pf);
		fseek(pf, 0L, SEEK_SET);
		pData = new unsigned char[size];
		fread(pData, size, 1, pf);
		fclose(pf);
	}
	if (pData)
	{
		int imageSize = width*height * 4;
		unsigned char* pImage = new unsigned char[imageSize];
		CoverbySSE2(width, height, 2048, pData, pImage);
		FILE* pf2 = fopen("d:\\testcuda_sse.rgb", "wb");
		if (pf2)
		{
			fwrite(pImage, imageSize, 1, pf2);
			fclose(pf2);
		}
		delete[] pImage;
	}
// 	uint32 ARGBpixel = 0;
// 	uint32 blue = 0x7;
// 	uint32 green = 1;
// 	uint32 red = 1;
// 	blue = blue >> 2;
// 	blue = blue << 8;
// 	uint8 chTemp = 0;
// 	chTemp = '1';
// 	ARGBpixel = (((uint32)blue) |
// 		(((uint32)green) << 8) |
// 		(((uint32)red) << 16));
// 	printf("ok!!!\n");
	return 0;
}