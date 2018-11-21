#pragma once

#include <stdio.h>
#include <conio.h>
#include <tchar.h>

#if _MSC_VER >=1400    // VC2005才支持intrin.h
#include <intrin.h>    // 所有Intrinsics函数
#else
#include <emmintrin.h>    // MMX, SSE, SSE2
#endif


// SSE系列指令集的支持级别. simd_sse_level 函数的返回值。
#define SIMD_SSE_NONE    0    // 不支持
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
// 64位下不支持内联汇编. 应使用__cpuid、__cpuidex等Intrinsics函数。
#else
#if _MSC_VER < 1600    // VS2010. 据说VC2008 SP1之后才支持__cpuidex
void __cpuidex(INT32 CPUInfo[4], INT32 InfoType, INT32 ECXValue)
{
	if (NULL == CPUInfo)    return;
	_asm {
		// load. 读取参数到寄存器
		mov edi, CPUInfo;    // 准备用edi寻址CPUInfo
		mov eax, InfoType;
		mov ecx, ECXValue;
		// CPUID
		cpuid;
		// save. 将寄存器保存到CPUInfo
		mov[edi], eax;
		mov[edi + 4], ebx;
		mov[edi + 8], ecx;
		mov[edi + 12], edx;
	}
}
#endif    // #if _MSC_VER < 1600    // VS2010. 据说VC2008 SP1之后才支持__cpuidex

#if _MSC_VER < 1400    // VC2005才支持__cpuid
void __cpuid(INT32 CPUInfo[4], INT32 InfoType)
{
	__cpuidex(CPUInfo, InfoType, 0);
}
#endif    // #if _MSC_VER < 1400    // VC2005才支持__cpuid

#endif    // #if defined(_WIN64)

// 取得CPU厂商（Vendor）
//
// result: 成功时返回字符串的长度（一般为12）。失败时返回0。
// pvendor: 接收厂商信息的字符串缓冲区。至少为13字节。
inline int cpu_getvendor(char* pvendor)
{
	int dwBuf[4];
	if (NULL == pvendor)    return 0;
	// Function 0: Vendor-ID and Largest Standard Function
	__cpuid(dwBuf, 0);
	// save. 保存到pvendor
	*(int*)&pvendor[0] = dwBuf[1];    // ebx: 前四个字符
	*(int*)&pvendor[4] = dwBuf[3];    // edx: 中间四个字符
	*(int*)&pvendor[8] = dwBuf[2];    // ecx: 最后四个字符
	pvendor[12] = '\0';
	return 12;
}

// 取得CPU商标（Brand）
//
// result: 成功时返回字符串的长度（一般为48）。失败时返回0。
// pbrand: 接收商标信息的字符串缓冲区。至少为49字节。
inline int cpu_getbrand(char* pbrand)
{
	int dwBuf[4];
	if (NULL == pbrand)    return 0;
	// Function 0x80000000: Largest Extended Function Number
	__cpuid(dwBuf, 0x80000000);
	if (dwBuf[0] < 0x80000004)    return 0;
	// Function 80000002h,80000003h,80000004h: Processor Brand String
	__cpuid((int*)&pbrand[0], 0x80000002);    // 前16个字符
	__cpuid((int*)&pbrand[16], 0x80000003);    // 中间16个字符
	__cpuid((int*)&pbrand[32], 0x80000004);    // 最后16个字符
	pbrand[48] = '\0';
	return 48;
}


// 是否支持MMX指令集
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
		// VC编译器不支持64位下的MMX。
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

// 检测SSE系列指令集的支持级别
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
		if (0 != *(int*)&xmm1)    rt = SIMD_SSE_NONE;    // 避免Release模式编译优化时剔除上一条语句
	}
	__except (1)
	{
		rt = SIMD_SSE_NONE;
	}

	return rt;
}
