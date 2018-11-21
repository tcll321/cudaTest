/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.  This source code is a "commercial item" as
* that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer software" and "commercial computer software
* documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*/

// This sample needs at least CUDA 5.5 and a GPU that has at least Compute Capability 2.0

// This sample demonstrates a simple image processing pipeline.
// First, a JPEG file is huffman decoded and inverse DCT transformed and dequantized.
// Then the different planes are resized. Finally, the resized image is quantized, forward
// DCT transformed and huffman encoded.

#include "cuda_functions.h"

#include <npp.h>
#include <cuda_runtime.h>
#include <Exceptions.h>

#include "Endianess.h"
#include <math.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <helper_string.h>
#include <helper_cuda.h>

using namespace std;

struct FrameHeader
{
    unsigned char nSamplePrecision;
    unsigned short nHeight;
    unsigned short nWidth;
    unsigned char nComponents;
    unsigned char aComponentIdentifier[3];
    unsigned char aSamplingFactors[3];
    unsigned char aQuantizationTableSelector[3];
};

struct ScanHeader
{
    unsigned char nComponents;
    unsigned char aComponentSelector[3];
    unsigned char aHuffmanTablesSelector[3];
    unsigned char nSs;
    unsigned char nSe;
    unsigned char nA;
};

struct QuantizationTable
{
    unsigned char nPrecisionAndIdentifier;
    unsigned char aTable[64];
};

struct HuffmanTable
{
    unsigned char nClassAndIdentifier;
    unsigned char aCodes[16];
    unsigned char aTable[256];
};


int DivUp(int x, int d)
{
    return (x + d - 1) / d;
}

template<typename T>
void writeAndAdvance(unsigned char *&pData, T nElement)
{
    writeBigEndian<T>(pData, nElement);
    pData += sizeof(T);
}

void writeMarker(unsigned char nMarker, unsigned char *&pData)
{
    *pData++ = 0x0ff;
    *pData++ = nMarker;
}

void writeJFIFTag(unsigned char *&pData)
{
    const char JFIF_TAG[] =
    {
        0x4a, 0x46, 0x49, 0x46, 0x00,
        0x01, 0x02,
        0x00,
        0x00, 0x01, 0x00, 0x01,
        0x00, 0x00
    };

    writeMarker(0x0e0, pData);
    writeAndAdvance<unsigned short>(pData, sizeof(JFIF_TAG) + sizeof(unsigned short));
    memcpy(pData, JFIF_TAG, sizeof(JFIF_TAG));
    pData += sizeof(JFIF_TAG);
}

void writeFrameHeader(const FrameHeader &header, unsigned char *&pData)
{
    unsigned char aTemp[128];
    unsigned char *pTemp = aTemp;

    writeAndAdvance<unsigned char>(pTemp, header.nSamplePrecision);
    writeAndAdvance<unsigned short>(pTemp, header.nHeight);
    writeAndAdvance<unsigned short>(pTemp, header.nWidth);
    writeAndAdvance<unsigned char>(pTemp, header.nComponents);

    for (int c=0; c<header.nComponents; ++c)
    {
        writeAndAdvance<unsigned char>(pTemp,header.aComponentIdentifier[c]);
        writeAndAdvance<unsigned char>(pTemp,header.aSamplingFactors[c]);
        writeAndAdvance<unsigned char>(pTemp,header.aQuantizationTableSelector[c]);
    }

    unsigned short nLength = (unsigned short)(pTemp - aTemp);

    writeMarker(0x0C0, pData);
    writeAndAdvance<unsigned short>(pData, nLength + 2);
    memcpy(pData, aTemp, nLength);
    pData += nLength;
}

void writeScanHeader(const ScanHeader &header, unsigned char *&pData)
{
    unsigned char aTemp[128];
    unsigned char *pTemp = aTemp;

    writeAndAdvance<unsigned char>(pTemp, header.nComponents);

    for (int c=0; c<header.nComponents; ++c)
    {
        writeAndAdvance<unsigned char>(pTemp,header.aComponentSelector[c]);
        writeAndAdvance<unsigned char>(pTemp,header.aHuffmanTablesSelector[c]);
    }

    writeAndAdvance<unsigned char>(pTemp,  header.nSs);
    writeAndAdvance<unsigned char>(pTemp,  header.nSe);
    writeAndAdvance<unsigned char>(pTemp,  header.nA);

    unsigned short nLength = (unsigned short)(pTemp - aTemp);

    writeMarker(0x0DA, pData);
    writeAndAdvance<unsigned short>(pData, nLength + 2);
    memcpy(pData, aTemp, nLength);
    pData += nLength;
}

void writeQuantizationTable(const QuantizationTable &table, unsigned char *&pData)
{
    writeMarker(0x0DB, pData);
    writeAndAdvance<unsigned short>(pData, sizeof(QuantizationTable) + 2);
    memcpy(pData, &table, sizeof(QuantizationTable));
    pData += sizeof(QuantizationTable);
}

void writeHuffmanTable(const HuffmanTable &table, unsigned char *&pData)
{
    writeMarker(0x0C4, pData);

    // Number of Codes for Bit Lengths [1..16]
    int nCodeCount = 0;

    for (int i = 0; i < 16; ++i)
    {
        nCodeCount += table.aCodes[i];
    }

    writeAndAdvance<unsigned short>(pData, 17 + nCodeCount + 2);
    memcpy(pData, &table, 17 + nCodeCount);
    pData += 17 + nCodeCount;
}

bool printfNPPinfo(int cudaVerMajor, int cudaVerMinor)
{
    const NppLibraryVersion *libVer   = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion/1000, (driverVersion%100)/10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);

    bool bVal = checkCudaCapabilities(cudaVerMajor, cudaVerMinor);
    return bVal;
}

// d_srcRGB数据排列形式为  BBBBBB......GGGGGGG......RRRRRRRR......
int jpegNPP(const char *szOutputFile, float* d_srcRGB, int img_width, int img_height)
{
    NppiDCTState *pDCTState;
    NPP_CHECK_NPP(nppiDCTInitAlloc(&pDCTState));

    // Parsing and Huffman Decoding (on host)
    FrameHeader oFrameHeader;
    QuantizationTable aQuantizationTables[4];
    Npp8u *pdQuantizationTables;
    cudaMalloc(&pdQuantizationTables, 64 * 4);

    HuffmanTable aHuffmanTables[4];
    HuffmanTable *pHuffmanDCTables = aHuffmanTables;
    HuffmanTable *pHuffmanACTables = &aHuffmanTables[2];
    ScanHeader oScanHeader;
    memset(&oFrameHeader,0,sizeof(FrameHeader));
    memset(aQuantizationTables,0, 4 * sizeof(QuantizationTable));
    memset(aHuffmanTables,0, 4 * sizeof(HuffmanTable));
    int nMCUBlocksH = 0;
    int nMCUBlocksV = 0;

    int nRestartInterval = -1;

    NppiSize aSrcSize[3];
    Npp16s *apdDCT[3] = {0,0,0};
    Npp32s aDCTStep[3];

    Npp8u *apSrcImage[3] = {0,0,0};
    Npp32s aSrcImageStep[3];
	size_t aSrcPitch[3];

    /***************************
    *
    *   Output    编码部分
    *
    ***************************/

	unsigned char STD_DC_Y_NRCODES[16] = { 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 };
	unsigned char STD_DC_Y_VALUES[12] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

	unsigned char STD_DC_UV_NRCODES[16] = { 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 };
	unsigned char STD_DC_UV_VALUES[12] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

	unsigned char STD_AC_Y_NRCODES[16] = { 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0X7D };
	unsigned char STD_AC_Y_VALUES[162] =
	{
		0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
		0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
		0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
		0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
		0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
		0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
		0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
		0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
		0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
		0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
		0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
		0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
		0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
		0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
		0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
		0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
		0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
		0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
		0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
		0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
		0xf9, 0xfa
	};

	unsigned char STD_AC_UV_NRCODES[16] = { 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0X77 };
	unsigned char STD_AC_UV_VALUES[162] =
	{
		0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
		0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
		0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
		0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
		0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
		0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
		0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
		0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
		0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
		0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
		0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
		0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
		0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
		0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
		0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
		0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
		0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
		0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
		0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
		0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
		0xf9, 0xfa
	};

	//填充Huffman表
	aHuffmanTables[0].nClassAndIdentifier = 0;
	memcpy(aHuffmanTables[0].aCodes, STD_DC_Y_NRCODES, 16);
	memcpy(aHuffmanTables[0].aTable, STD_DC_Y_VALUES, 12);

	aHuffmanTables[1].nClassAndIdentifier = 1;
	memcpy(aHuffmanTables[1].aCodes, STD_DC_UV_NRCODES, 16);
	memcpy(aHuffmanTables[1].aTable, STD_DC_UV_VALUES, 12);

	aHuffmanTables[2].nClassAndIdentifier = 16;
	memcpy(aHuffmanTables[2].aCodes, STD_AC_Y_NRCODES, 16);
	memcpy(aHuffmanTables[2].aTable, STD_AC_Y_VALUES, 162);

	aHuffmanTables[3].nClassAndIdentifier = 17;
	memcpy(aHuffmanTables[3].aCodes, STD_AC_UV_NRCODES, 16);
	memcpy(aHuffmanTables[3].aTable, STD_AC_UV_VALUES, 162);

	//标准亮度信号量化模板
	unsigned char std_Y_QT[64] =
	{
		16, 11, 10, 16, 24, 40, 51, 61,
		12, 12, 14, 19, 26, 58, 60, 55,
		14, 13, 16, 24, 40, 57, 69, 56,
		14, 17, 22, 29, 51, 87, 80, 62,
		18, 22, 37, 56, 68, 109, 103, 77,
		24, 35, 55, 64, 81, 104, 113, 92,
		49, 64, 78, 87, 103, 121, 120, 101,
		72, 92, 95, 98, 112, 100, 103, 99
	};

	//标准色差信号量化模板
	unsigned char std_UV_QT[64] =
	{
		17, 18, 24, 47, 99, 99, 99, 99,
		18, 21, 26, 66, 99, 99, 99, 99,
		24, 26, 56, 99, 99, 99, 99, 99,
		47, 66, 99, 99, 99, 99, 99, 99,
		99, 99, 99, 99, 99, 99, 99, 99,
		99, 99, 99, 99, 99, 99, 99, 99,
		99, 99, 99, 99, 99, 99, 99, 99,
		99, 99, 99, 99, 99, 99, 99, 99
	};

	////亮度信号量化模板
	//unsigned char std_Y_QT[64] =
	//{
	//	6, 4, 5, 6, 5, 4, 6, 6,
	//	5, 6, 7, 7, 6, 8, 10, 16,
	//	10, 10, 9, 9, 10, 20, 14, 15,
	//	12, 16, 23, 20, 24, 24, 23, 20,
	//	22, 22, 26, 29, 37, 31, 26, 27,
	//	35, 28, 22, 22, 32, 44, 32, 35,
	//	38, 39, 41, 42, 41, 25, 31, 45,
	//	48, 45, 40, 48, 37, 40, 41, 40
	//};

	////色差信号量化模板
	//unsigned char std_UV_QT[64] =
	//{
	//	7, 7, 7, 10, 8, 10, 19, 10,
	//	10, 19, 40, 26, 22, 26, 40, 40,
	//	40, 40, 40, 40, 40, 40, 40, 40,
	//	40, 40, 40, 40, 40, 40, 40, 40,
	//	40, 40, 40, 40, 40, 40, 40, 40,
	//	40, 40, 40, 40, 40, 40, 40, 40,
	//	40, 40, 40, 40, 40, 40, 40, 40,
	//	40, 40, 40, 40, 40, 40, 40, 40
	//};

	//填充量化表
	aQuantizationTables[0].nPrecisionAndIdentifier = 0;
	memcpy(aQuantizationTables[0].aTable, std_Y_QT, 64);
	aQuantizationTables[1].nPrecisionAndIdentifier = 1;
	memcpy(aQuantizationTables[1].aTable, std_UV_QT, 64);

	NPP_CHECK_CUDA(cudaMemcpyAsync(pdQuantizationTables , aQuantizationTables[0].aTable, 64, cudaMemcpyHostToDevice));
	NPP_CHECK_CUDA(cudaMemcpyAsync(pdQuantizationTables + 64, aQuantizationTables[1].aTable, 64, cudaMemcpyHostToDevice));

	//填充帧头
	oFrameHeader.nSamplePrecision = 8;
	oFrameHeader.nComponents = 3;
	oFrameHeader.aComponentIdentifier[0] = 1;
	oFrameHeader.aComponentIdentifier[1] = 2;
	oFrameHeader.aComponentIdentifier[2] = 3;
	oFrameHeader.aSamplingFactors[0] = 34;
	oFrameHeader.aSamplingFactors[1] = 17;
	oFrameHeader.aSamplingFactors[2] = 17;
	oFrameHeader.aQuantizationTableSelector[0] = 0;
	oFrameHeader.aQuantizationTableSelector[1] = 1;
	oFrameHeader.aQuantizationTableSelector[2] = 1;
	oFrameHeader.nWidth = img_width;
	oFrameHeader.nHeight = img_height;

	for (int i = 0; i < oFrameHeader.nComponents; ++i)
	{
		nMCUBlocksV = max(nMCUBlocksV, oFrameHeader.aSamplingFactors[i] & 0x0f);
		nMCUBlocksH = max(nMCUBlocksH, oFrameHeader.aSamplingFactors[i] >> 4);
	}

	for (int i = 0; i < oFrameHeader.nComponents; ++i)
	{
		NppiSize oBlocks;
		NppiSize oBlocksPerMCU = { oFrameHeader.aSamplingFactors[i] >> 4, oFrameHeader.aSamplingFactors[i] & 0x0f };

		oBlocks.width = (int)ceil((oFrameHeader.nWidth + 7) / 8 *
			static_cast<float>(oBlocksPerMCU.width) / nMCUBlocksH);
		oBlocks.width = DivUp(oBlocks.width, oBlocksPerMCU.width) * oBlocksPerMCU.width;

		oBlocks.height = (int)ceil((oFrameHeader.nHeight + 7) / 8 *
			static_cast<float>(oBlocksPerMCU.height) / nMCUBlocksV);
		oBlocks.height = DivUp(oBlocks.height, oBlocksPerMCU.height) * oBlocksPerMCU.height;

		aSrcSize[i].width = oBlocks.width * 8;
		aSrcSize[i].height = oBlocks.height * 8;

		// Allocate Memory
		size_t nPitch;
		NPP_CHECK_CUDA(cudaMallocPitch(&apdDCT[i], &nPitch, oBlocks.width * 64 * sizeof(Npp16s), oBlocks.height));
		aDCTStep[i] = static_cast<Npp32s>(nPitch);

		NPP_CHECK_CUDA(cudaMallocPitch(&apSrcImage[i], &nPitch, aSrcSize[i].width, aSrcSize[i].height));

		aSrcPitch[i] = nPitch;
		aSrcImageStep[i] = static_cast<Npp32s>(nPitch);
	}

	//RGB2YUV
	cudaError_t cudaStatus;
	cudaStatus = cuda_common::RGB2YUV(d_srcRGB, img_width, img_height,
										apSrcImage[0], aSrcPitch[0], aSrcSize[0].width, aSrcSize[0].height,
										apSrcImage[1], aSrcPitch[1], aSrcSize[1].width, aSrcSize[1].height,
										apSrcImage[2], aSrcPitch[2], aSrcSize[2].width, aSrcSize[2].height);

	/**
	* Forward DCT, quantization and level shift part of the JPEG encoding.
	* Input is expected in 8x8 macro blocks and output is expected to be in 64x1
	* macro blocks. The new version of the primitive takes the ROI in image pixel size and
	* works with DCT coefficients that are in zig-zag order.
	*/
	int k = 0;
	NPP_CHECK_NPP(nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R_NEW(apSrcImage[0], aSrcImageStep[0],
															apdDCT[0], aDCTStep[0],
															pdQuantizationTables + k * 64,
															aSrcSize[0],
															pDCTState));
	k = 1;
	NPP_CHECK_NPP(nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R_NEW(apSrcImage[1], aSrcImageStep[1],
															apdDCT[1], aDCTStep[1],
															pdQuantizationTables + k * 64,
															aSrcSize[1],
															pDCTState));

	NPP_CHECK_NPP(nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R_NEW(apSrcImage[2], aSrcImageStep[2],
															apdDCT[2], aDCTStep[2],
															pdQuantizationTables + k * 64,
															aSrcSize[2],
															pDCTState));


    // Huffman Encoding
    Npp8u *pdScan;
    Npp32s nScanLength;
    NPP_CHECK_CUDA(cudaMalloc(&pdScan, 4 << 20));

    Npp8u *pJpegEncoderTemp;
    Npp32s nTempSize;
    NPP_CHECK_NPP(nppiEncodeHuffmanGetSize(aSrcSize[0], 3, (size_t*)&nTempSize));
    NPP_CHECK_CUDA(cudaMalloc(&pJpegEncoderTemp, nTempSize));

    NppiEncodeHuffmanSpec *apHuffmanDCTable[3];
    NppiEncodeHuffmanSpec *apHuffmanACTable[3];

	/**
	* Allocates memory and creates a Huffman table in a format that is suitable for the encoder.
	*/
	NppStatus t_status ;
	t_status = nppiEncodeHuffmanSpecInitAlloc_JPEG(pHuffmanDCTables[0].aCodes, nppiDCTable, &apHuffmanDCTable[0]);
	t_status = nppiEncodeHuffmanSpecInitAlloc_JPEG(pHuffmanACTables[0].aCodes, nppiACTable, &apHuffmanACTable[0]);
	t_status = nppiEncodeHuffmanSpecInitAlloc_JPEG(pHuffmanDCTables[1].aCodes, nppiDCTable, &apHuffmanDCTable[1]);
	t_status = nppiEncodeHuffmanSpecInitAlloc_JPEG(pHuffmanACTables[1].aCodes, nppiACTable, &apHuffmanACTable[1]);
	t_status = nppiEncodeHuffmanSpecInitAlloc_JPEG(pHuffmanDCTables[1].aCodes, nppiDCTable, &apHuffmanDCTable[2]);
	t_status = nppiEncodeHuffmanSpecInitAlloc_JPEG(pHuffmanACTables[1].aCodes, nppiACTable, &apHuffmanACTable[2]);

	/**
	* Huffman Encoding of the JPEG Encoding.
	* Input is expected to be 64x1 macro blocks and output is expected as byte stuffed huffman encoded JPEG scan.
	*/
	Npp32s nSs = 0;
	Npp32s nSe = 63;
	Npp32s nH = 0;
	Npp32s nL = 0;
    NPP_CHECK_NPP(nppiEncodeHuffmanScan_JPEG_8u16s_P3R(apdDCT, aDCTStep,
													   0, nSs, nSe, nH, nL,
                                                       pdScan, &nScanLength,
                                                       apHuffmanDCTable,
                                                       apHuffmanACTable,
													   aSrcSize,
                                                       pJpegEncoderTemp));

    for (int i = 0; i < 3; ++i)
    {
        nppiEncodeHuffmanSpecFree_JPEG(apHuffmanDCTable[i]);
        nppiEncodeHuffmanSpecFree_JPEG(apHuffmanACTable[i]);
    }

    // Write JPEG
    unsigned char *pDstJpeg = new unsigned char[4 << 20];
    unsigned char *pDstOutput = pDstJpeg;

    writeMarker(0x0D8, pDstOutput);
    writeJFIFTag(pDstOutput);
	writeQuantizationTable(aQuantizationTables[0], pDstOutput);
	writeQuantizationTable(aQuantizationTables[1], pDstOutput);

    writeFrameHeader(oFrameHeader, pDstOutput);
    writeHuffmanTable(pHuffmanDCTables[0], pDstOutput);
    writeHuffmanTable(pHuffmanACTables[0], pDstOutput);
    writeHuffmanTable(pHuffmanDCTables[1], pDstOutput);
    writeHuffmanTable(pHuffmanACTables[1], pDstOutput);

	oScanHeader.nComponents = 3;
	oScanHeader.aComponentSelector[0] = 1;
	oScanHeader.aComponentSelector[1] = 2;
	oScanHeader.aComponentSelector[2] = 3;
	oScanHeader.aHuffmanTablesSelector[0] = 0;
	oScanHeader.aHuffmanTablesSelector[1] = 17;
	oScanHeader.aHuffmanTablesSelector[2] = 17;
	oScanHeader.nSs = 0;
	oScanHeader.nSe = 63;
	oScanHeader.nA = 0;

    writeScanHeader(oScanHeader, pDstOutput);
    NPP_CHECK_CUDA(cudaMemcpy(pDstOutput, pdScan, nScanLength, cudaMemcpyDeviceToHost));
    pDstOutput += nScanLength;
    writeMarker(0x0D9, pDstOutput);

    {
        // Write result to file.
        std::ofstream outputFile(szOutputFile, ios::out | ios::binary);
        outputFile.write(reinterpret_cast<const char *>(pDstJpeg), static_cast<int>(pDstOutput - pDstJpeg));
    }

    // Cleanup
    delete [] pDstJpeg;

    cudaFree(pJpegEncoderTemp);
    cudaFree(pdQuantizationTables);
    cudaFree(pdScan);

    nppiDCTFree(pDCTState);

    for (int i = 0; i < 3; ++i)
    {
        cudaFree(apdDCT[i]);
        cudaFree(apSrcImage[i]);
    }

    return EXIT_SUCCESS;
}
