#pragma once

#include <stdio.h>
#include <conio.h>
#include <tchar.h>

#if _MSC_VER >=1400    // VC2005��֧��intrin.h
#include <intrin.h>    // ����Intrinsics����
#else
#include <emmintrin.h>    // MMX, SSE, SSE2
#endif


// SSEϵ��ָ���֧�ּ���. simd_sse_level �����ķ���ֵ��
#define SIMD_SSE_NONE    0    // ��֧��
#define SIMD_SSE_1    1    // SSE
#define SIMD_SSE_2    2    // SSE2
#define SIMD_SSE_3    3    // SSE3
#define SIMD_SSE_3S    4    // SSSE3
#define SIMD_SSE_41    5    // SSE4.1
#define SIMD_SSE_42    6    // SSE4.2

const char*    simd_sse_names[] = {
	"None",
	"SSE",
	"SSE2",
	"SSE3",
	"SSSE3",
	"SSE4.1",
	"SSE4.2",
};

char szBuf[64];
int dwBuf[4];
int cpu_getvendor(char* pvendor);
int cpu_getbrand(char* pbrand);
bool simd_mmx(bool* phwmmx);
int  simd_sse_level(int* phwsse);

#if defined(_WIN64)
// 64λ�²�֧���������. Ӧʹ��__cpuid��__cpuidex��Intrinsics������
#else
#if _MSC_VER < 1600    // VS2010. ��˵VC2008 SP1֮���֧��__cpuidex
void __cpuidex(INT32 CPUInfo[4], INT32 InfoType, INT32 ECXValue)
{
	if (NULL == CPUInfo)    return;
	_asm {
		// load. ��ȡ�������Ĵ���
		mov edi, CPUInfo;    // ׼����ediѰַCPUInfo
		mov eax, InfoType;
		mov ecx, ECXValue;
		// CPUID
		cpuid;
		// save. ���Ĵ������浽CPUInfo
		mov[edi], eax;
		mov[edi + 4], ebx;
		mov[edi + 8], ecx;
		mov[edi + 12], edx;
	}
}
#endif    // #if _MSC_VER < 1600    // VS2010. ��˵VC2008 SP1֮���֧��__cpuidex

#if _MSC_VER < 1400    // VC2005��֧��__cpuid
void __cpuid(INT32 CPUInfo[4], INT32 InfoType)
{
	__cpuidex(CPUInfo, InfoType, 0);
}
#endif    // #if _MSC_VER < 1400    // VC2005��֧��__cpuid

#endif    // #if defined(_WIN64)

// ȡ��CPU���̣�Vendor��
//
// result: �ɹ�ʱ�����ַ����ĳ��ȣ�һ��Ϊ12����ʧ��ʱ����0��
// pvendor: ���ճ�����Ϣ���ַ���������������Ϊ13�ֽڡ�
inline int cpu_getvendor(char* pvendor)
{
	int dwBuf[4];
	if (NULL == pvendor)    return 0;
	// Function 0: Vendor-ID and Largest Standard Function
	__cpuid(dwBuf, 0);
	// save. ���浽pvendor
	*(int*)&pvendor[0] = dwBuf[1];    // ebx: ǰ�ĸ��ַ�
	*(int*)&pvendor[4] = dwBuf[3];    // edx: �м��ĸ��ַ�
	*(int*)&pvendor[8] = dwBuf[2];    // ecx: ����ĸ��ַ�
	pvendor[12] = '\0';
	return 12;
}

// ȡ��CPU�̱꣨Brand��
//
// result: �ɹ�ʱ�����ַ����ĳ��ȣ�һ��Ϊ48����ʧ��ʱ����0��
// pbrand: �����̱���Ϣ���ַ���������������Ϊ49�ֽڡ�
inline int cpu_getbrand(char* pbrand)
{
	int dwBuf[4];
	if (NULL == pbrand)    return 0;
	// Function 0x80000000: Largest Extended Function Number
	__cpuid(dwBuf, 0x80000000);
	if (dwBuf[0] < 0x80000004)    return 0;
	// Function 80000002h,80000003h,80000004h: Processor Brand String
	__cpuid((int*)&pbrand[0], 0x80000002);    // ǰ16���ַ�
	__cpuid((int*)&pbrand[16], 0x80000003);    // �м�16���ַ�
	__cpuid((int*)&pbrand[32], 0x80000004);    // ���16���ַ�
	pbrand[48] = '\0';
	return 48;
}


// �Ƿ�֧��MMXָ�
inline bool simd_mmx(bool* phwmmx)
{
	const int    BIT_D_MMX = 0x00800000;    // bit 23
	bool    rt = false;    // result
	int dwBuf[4];

	// check processor support
	__cpuid(dwBuf, 1);    // Function 1: Feature Information
	if (dwBuf[3] & BIT_D_MMX)    rt = true;
	if (NULL != phwmmx)    *phwmmx = rt;

	// check OS support
	if (rt)
	{
#if defined(_WIN64)
		// VC��������֧��64λ�µ�MMX��
		rt = false;
#else
		__try
		{
			_mm_empty();    // MMX instruction: emms
		}
		__except (EXCEPTION_EXECUTE_HANDLER)
		{
			rt = FALSE;
		}
#endif    // #if defined(_WIN64)
	}

	return rt;
}

// ���SSEϵ��ָ���֧�ּ���
inline int  simd_sse_level(int* phwsse)
{
	const int    BIT_D_SSE = 0x02000000;    // bit 25
	const int    BIT_D_SSE2 = 0x04000000;    // bit 26
	const int    BIT_C_SSE3 = 0x00000001;    // bit 0
	const int    BIT_C_SSSE3 = 0x00000100;    // bit 9
	const int    BIT_C_SSE41 = 0x00080000;    // bit 19
	const int    BIT_C_SSE42 = 0x00100000;    // bit 20
	int    rt = SIMD_SSE_NONE;    // result
	int dwBuf[4];

	// check processor support
	__cpuid(dwBuf, 1);    // Function 1: Feature Information
	if (dwBuf[3] & BIT_D_SSE)
	{
		rt = SIMD_SSE_1;
		if (dwBuf[3] & BIT_D_SSE2)
		{
			rt = SIMD_SSE_2;
			if (dwBuf[2] & BIT_C_SSE3)
			{
				rt = SIMD_SSE_3;
				if (dwBuf[2] & BIT_C_SSSE3)
				{
					rt = SIMD_SSE_3S;
					if (dwBuf[2] & BIT_C_SSE41)
					{
						rt = SIMD_SSE_41;
						if (dwBuf[2] & BIT_C_SSE42)
						{
							rt = SIMD_SSE_42;
						}
					}
				}
			}
		}
	}
	if (NULL != phwsse)    *phwsse = rt;

	// check OS support
	__try
	{
		__m128 xmm1 = _mm_setzero_ps();    // SSE instruction: xorps
		if (0 != *(int*)&xmm1)    rt = SIMD_SSE_NONE;    // ����Releaseģʽ�����Ż�ʱ�޳���һ�����
	}
	__except (1)
	{
		rt = SIMD_SSE_NONE;
	}

	return rt;
}
