#pragma once
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"

using namespace cv;

class ImageFactory
{
public:
	ImageFactory(void);

	//����ͼ�񣬲����ö��ձ������� Mat ͼ��
	Mat ReadImage(const char* path, const ImreadModes modes);

	//չʾͼ��
	void ShowImage(const Mat input);

	//ʵ���˲������� Mat ͼ��
	Mat Filter(const Mat input, const String mode);

	//ʵ�����Ա任��alpha��б�ʣ�beta�ǲ���
	Mat GrayTrans(const Mat input, const double alpha, const double beta);

	//������MatͼƬ����ֱ��ͼƥ�䣬����ƥ��̶�
	double CompareHist(const Mat image1, const Mat image2);

	//����"״̬��"��"�жϷ�����"���ַ���ʵ�ֶ�ֵ�������� Mat ͼ��
	Mat Binarization(const Mat input, const String mode);

	//ͼ��ƽ��
	Mat Translation(const Mat input, const double x, double y);

	//ͼ������
	Mat TransScale(const Mat  input, const double x, const double y);

	//ͼ����ת
	Mat TransRotate(Mat const input, double theta);
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

