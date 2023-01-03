#pragma once
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"

using namespace cv;

class ImageFactory
{
public:
	ImageFactory(void);

	//����ͼ�񣬲����ö��ձ������� Mat ͼ��
	ImageFactory ReadImage(const char* path, const ImreadModes modes);

	//չʾͼ��
	void ShowImage();

	//ʵ���˲�
	ImageFactory Filter(const String mode);

	//ʵ�����Ա任��alpha��б�ʣ�beta�ǲ���
	ImageFactory GrayTrans(const double alpha, const double beta);

	//������MatͼƬ����ֱ��ͼƥ�䣬����ƥ��̶�
	static double CompareHist(const ImageFactory image1, const ImageFactory image2);

	//����"״̬��"��"�жϷ�����"���ַ���ʵ�ֶ�ֵ��
	ImageFactory Binarization(const String mode);

	//ͼ��ƽ��
	ImageFactory Translation(const double x, const double y);

	//ͼ������
	ImageFactory TransScale(const double x, const double y);

	//ͼ����ת
	ImageFactory TransRotate(double theta);

	//�õ��ڲ������ָ��
	Mat* getMatrix(void);

	//ɫ��ƽ��
	ImageFactory ColorBalance(void);

private:
	Mat matrixCopy;

	//�������������ԭͼ��;���ˣ����� Mat
	Mat Convolution(const Mat input, const double* kernel);

	//��ֵ�˲�������ֻд��3*3�Ĵ��ڣ����� Mat
	Mat Median(const Mat input);

	//�����䷽���ֵ��
	Mat OtsuThresh(const Mat SrcImg);

	//״̬����ֵ��
	Mat StateThresh(const Mat SrcImg);
};

