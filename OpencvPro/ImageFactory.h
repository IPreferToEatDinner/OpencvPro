#pragma once
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"

using namespace cv;

class ImageFactory
{
public:
	ImageFactory(void);

	//读入图像，并做好读空报错，返回 Mat 图像
	Mat ReadImage(const char* path, const ImreadModes modes);

	//展示图像
	void ShowImage(const Mat input);

	//实现滤波，返回 Mat 图像
	Mat Filter(const Mat input, const String mode);

	//实现线性变换，alpha是斜率，beta是参数
	Mat GrayTrans(const Mat input, const double alpha, const double beta);

	//将两张Mat图片进行直方图匹配，返回匹配程度
	double CompareHist(const Mat image1, const Mat image2);

	//利用"状态法"、"判断分析法"两种方法实现二值化，返回 Mat 图像
	Mat Binarization(const Mat input, const String mode);

	//图像平移
	Mat Translation(const Mat input, const double x, double y);

	//图像缩放
	Mat TransScale(const Mat  input, const double x, const double y);

	//图像旋转
	Mat TransRotate(Mat const input, double theta);
private:
	Mat matrixCopy;

	//卷积函数，传入原图像和卷积核，返回 Mat
	Mat Convolution(const Mat input, const double* kernel);

	//中值滤波函数，只写了3*3的窗口，返回 Mat
	Mat Median(const Mat input);

	//最大类间方差法二值化
	Mat OtsuThresh(const Mat SrcImg);

	//状态法二值化
	Mat StateThresh(const Mat SrcImg);
};

