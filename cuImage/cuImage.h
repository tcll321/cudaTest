// ���� ifdef ���Ǵ���ʹ�� DLL �������򵥵�
// ��ı�׼�������� DLL �е������ļ��������������϶���� CUIMAGE_EXPORTS
// ���ű���ġ���ʹ�ô� DLL ��
// �κ�������Ŀ�ϲ�Ӧ����˷��š�������Դ�ļ��а������ļ����κ�������Ŀ���Ὣ
// CUIMAGE_API ������Ϊ�Ǵ� DLL ����ģ����� DLL ���ô˺궨���
// ������Ϊ�Ǳ������ġ�
#ifdef CUIMAGE_EXPORTS
#define CUIMAGE_API __declspec(dllexport)
#else
#define CUIMAGE_API __declspec(dllimport)
#endif

// �����Ǵ� cuImage.dll ������
class CUIMAGE_API CcuImage {
public:
	CcuImage(void);
	// TODO:  �ڴ�������ķ�����
	int Init(int deviceID);
	//	srcImage GPU�ڴ�RGB����
	//	dstImage �ڴ�RGB����
	virtual int MattingImage(void* srcImage, int srcWidth, int srcHeight, BYTE* dstImage, int x, int y, int dstWidth, int dstHeight);
};

extern CUIMAGE_API int ncuImage;

CUIMAGE_API int fncuImage(void);
