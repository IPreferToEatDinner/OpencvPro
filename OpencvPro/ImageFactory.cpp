#include "ImageFactory.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


ImageFactory::ImageFactory()
{
	std::cout << "此为工厂函数，遵循函数式编程原则，不会改变原矩阵，同时返回改变后的矩阵" << endl;
}

Mat ImageFactory::ReadImage(const char* path, const ImreadModes modes)
{
	Mat output = imread(path, modes);

	// 判断文件是否正常打开  
	if (output.empty())
	{
		fprintf(stderr, "Can not load image %s\n", path);
		waitKey(5000);  // 等待6000 ms后窗口自动关闭   
		exit(0);
	}

	return output;
}

void ImageFactory::ShowImage(const Mat input)
{
	imshow("image", input);
	waitKey(0);
}

Mat ImageFactory::Filter(const Mat input, const String mode)
{
	double laplacian[9] = {
		0, -1, 0,
		-1, 4, -1,
		0, -1, 0 };

	double average[9] = {
		1 / 9.0, 1 / 9.0, 1 / 9.0,
		1 / 9.0, 1 / 9.0, 1 / 9.0,
		1 / 9.0, 1 / 9.0, 1 / 9.0 };

	if (mode == "高通滤波") {
		return this->Convolution(input, laplacian);
	}
	else if (mode == "低通滤波")
	{
		return	this->Convolution(input, average);
	}
	else if (mode == "中值滤波") {
		return this->Median(input);
	}
	else {
		cerr << "滤波模式选择错误，没有这种滤波模式" << endl;
		exit(0);
	}


	return this->matrixCopy;
}

Mat ImageFactory::GrayTrans(const Mat input, const double alpha, const double beta)
{
	//获取原图像的值和其他参数
	unsigned char* pSrc = input.data;
	int rows = input.rows;
	int cols = input.cols;
	int nchannels = input.channels();

	Mat DstImg;
	//如果不是 1 或者 3 通道就直接挂
	nchannels == 1 ? DstImg.create(rows, cols, CV_8UC1) :
		nchannels == 3 ? DstImg.create(rows, cols, CV_8UC3) : exit(0);

	unsigned char* pDst = DstImg.data;
	int record;

	for (int i = 0; i < nchannels; i++)
	{
		for (int j = 0; j < rows; j++)
		{
			for (int l = 0; l < cols; l++)
			{
				record = alpha * pSrc[(j * cols + l) * nchannels + i] + beta;

				if (record > 255) {
					pDst[(j * cols + l) * nchannels + i] = 255;
				}
				else if (record < 0) {
					pDst[(j * cols + l) * nchannels + i] = 0;
				}
				else {
					pDst[(j * cols + l) * nchannels + i] = record;
				}
			}
		}
	}

	return DstImg;
}

double ImageFactory::CompareHist(const Mat image1, const Mat image2)
{
	// 计算模板图像的直方图
	Mat hist1;
	int channels[] = { 0 };
	int histSize[] = { 256 };
	float range[] = { 0, 256 };
	const float* ranges[] = { range };
	calcHist(&image1, 1, channels, Mat(), hist1, 1, histSize, ranges, true, false);

	// 规格化直方图
	normalize(hist1, hist1, 0, 1, NORM_MINMAX, -1, Mat());

	// 计算待匹配图像的直方图
	Mat hist2;
	calcHist(&image2, 1, channels, Mat(), hist2, 1, histSize, ranges, true, false);

	// 规格化直方图
	normalize(hist2, hist2, 0, 1, NORM_MINMAX, -1, Mat());

	// 进行直方图匹配
	return compareHist(hist1, hist2, HISTCMP_CORREL);
}

Mat ImageFactory::Binarization(const Mat input, const String mode)
{
	Mat gray;
	cvtColor(input, gray, COLOR_BGR2GRAY);

	if (mode == "状态法") {
		//int limit = 128;
		//threshold(gray, this->matrixCopy, limit, 255, cv::THRESH_BINARY);
		//return this->matrixCopy;

		return this->StateThresh(gray);
	}
	else if (mode == "判断分析法") {
		return this->OtsuThresh(gray);
	}
	else
	{
		cerr << "输入的二值化模式不存在" << endl;
		exit(0);
	}
}

Mat ImageFactory::Translation(const Mat input, const double x, double y)
{
	int rows = input.rows;
	int cols = input.cols;
	int nchannels = input.channels();

	Mat DstImg;
	//输出图像初始化
	if (nchannels == 1) {
		DstImg = Mat::zeros(rows, cols, CV_8UC1);
	}
	else if (nchannels == 3) {
		DstImg = Mat::zeros(rows, cols, CV_8UC3);
	}

	uchar* pSrc = input.data;
	uchar* pDst = DstImg.data;

	//构造平移变换矩阵
	Mat T = (Mat_<double>(3, 3) <<
		1, 0, 0,
		0, 1, 0,
		x, y, 1);

	Mat T_inv = T.inv();//求逆

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			Mat dst_xy = (Mat_<double>(1, 3) << j, i, 1);//第i行j列的像素坐标(j,i)
			Mat src_uv = dst_xy * T_inv;

			double u = src_uv.at<double>(0, 0);//原图像的横坐标，对应图像的列数
			double v = src_uv.at<double>(0, 1);//原图像的纵坐标，对应图像的行数

			//双线性插值法
			if (u >= 0 && v >= 0 && u <= cols - 1 && v <= rows - 1) {//判断对应的(u,v)是否在原图像范围内，即是否越界
				int top = floor(v), bottom = ceil(v), left = floor(u), right = ceil(u);
				double dv = v - top, du = u - left;//坐标偏差值，即小数部分

				for (int k = 0; k < nchannels; k++) {
					pDst[(i * cols + j) * nchannels + k] = (1 - dv) * (1 - du) * pSrc[(top * input.cols + left) * nchannels + k] + (1 - dv) * du * pSrc[(top * input.cols + right) * nchannels + k] + dv * (1 - du) * pSrc[(bottom * input.cols + left) * nchannels + k] + dv * du * pSrc[(bottom * input.cols + right) * nchannels + k];
				}
			}

		}
	}

	return DstImg;
}

Mat ImageFactory::TransScale(const Mat input, const double x, const double y)
{
	int rows = round(input.rows * x);
	int cols = round(input.cols * y);
	int nchannels = input.channels();

	Mat DstImg;
	//输出图像初始化
	if (nchannels == 1) {
		DstImg = Mat::zeros(rows, cols, CV_8UC1);
	}
	else if (nchannels == 3) {
		DstImg = Mat::zeros(rows, cols, CV_8UC3);
	}

	uchar* pSrc = input.data;
	uchar* pDst = DstImg.data;

	//构造缩放变换矩阵
	Mat T = (Mat_<double>(3, 3) << x, 0, 0, 0, y, 0, 0, 0, 1);
	Mat T_inv = T.inv();//求逆

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			Mat dst_xy = (Mat_<double>(1, 3) << j, i, 1);//第i行j列的像素坐标(j,i)
			Mat src_uv = dst_xy * T_inv;

			double u = src_uv.at<double>(0, 0);//原图像的横坐标，对应图像的列数
			double v = src_uv.at<double>(0, 1);//原图像的纵坐标，对应图像的行数

			//双线性插值法
			if (u >= 0 && v >= 0 && u <= input.cols - 1 && v <= input.rows - 1) {
				int top = floor(v), bottom = ceil(v), left = floor(u), right = ceil(u); //与映射到原图坐标相邻的四个像素点的坐标
				double dv = v - top; //dv为坐标 行 的小数部分(坐标偏差)
				double du = u - left; //du为坐标 列 的小数部分(坐标偏差)

				for (int k = 0; k < nchannels; k++) {
					pDst[(i * cols + j) * nchannels + k] = (1 - dv) * (1 - du) * pSrc[(top * input.cols + left) * nchannels + k] + (1 - dv) * du * pSrc[(top * input.cols + right) * nchannels + k] + dv * (1 - du) * pSrc[(bottom * input.cols + left) * nchannels + k] + dv * du * pSrc[(bottom * input.cols + right) * nchannels + k];
				}

			}

		}
	}

	return DstImg;
}

Mat ImageFactory::TransRotate(Mat const input, double theta)
{
	theta = theta * CV_PI / 180;
	int rows = round(fabs(input.rows * cos(theta)) + fabs(input.cols * sin(theta)));
	int cols = round(fabs(input.cols * cos(theta)) + fabs(input.rows * sin(theta)));
	int nchannels = input.channels();

	Mat DstImg;
	//输出图像初始化
	if (nchannels == 1) {
		DstImg = Mat::zeros(rows, cols, CV_8UC1);
	}
	else if (nchannels == 3) {
		DstImg = Mat::zeros(rows, cols, CV_8UC3);
	}

	uchar* pSrc = input.data;
	uchar* pDst = DstImg.data;

	//构造旋转变换矩阵
	Mat T1 = (Mat_<double>(3, 3) << 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, -0.5 * input.cols, 0.5 * input.rows, 1.0);
	Mat T2 = (Mat_<double>(3, 3) << cos(theta), -sin(theta), 0.0, sin(theta), cos(theta), 0.0, 0.0, 0.0, 1.0);
	double t3[3][3] = { { 1.0, 0.0, 0.0 },{ 0.0, -1.0, 0.0 },{ 0.5 * DstImg.cols, 0.5 * DstImg.rows ,1.0 } }; // 将数学笛卡尔坐标映射到旋转后的图像坐标
	Mat T3 = Mat(3.0, 3.0, CV_64FC1, t3);

	Mat T = T1 * T2 * T3;
	Mat T_inv = T.inv();//求逆

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			Mat dst_xy = (Mat_<double>(1, 3) << j, i, 1);//第i行j列的像素坐标(j,i)
			Mat src_uv = dst_xy * T_inv;

			double u = src_uv.at<double>(0, 0);//原图像的横坐标，对应图像的列数
			double v = src_uv.at<double>(0, 1);//原图像的纵坐标，对应图像的行数

			//双线性插值法
			if (u >= 0 && v >= 0 && u <= input.cols - 1 && v <= input.rows - 1) {
				int top = floor(v), bottom = ceil(v), left = floor(u), right = ceil(u); //与映射到原图坐标相邻的四个像素点的坐标
				double dv = v - top; //dv为坐标 行 的小数部分(坐标偏差)
				double du = u - left; //du为坐标 列 的小数部分(坐标偏差)

				for (int k = 0; k < nchannels; k++) {
					pDst[(i * cols + j) * nchannels + k] = (1 - dv) * (1 - du) * pSrc[(top * input.cols + left) * nchannels + k] + (1 - dv) * du * pSrc[(top * input.cols + right) * nchannels + k] + dv * (1 - du) * pSrc[(bottom * input.cols + left) * nchannels + k] + dv * du * pSrc[(bottom * input.cols + right) * nchannels + k];
				}
			}


		}
	}

	return DstImg;
}

Mat ImageFactory::Convolution(const Mat input, const double* kernel)
{
	//获得原始图像的内容和参数
	unsigned char* pSrc = input.data;
	int rows = input.rows;
	int cols = input.cols;
	int nchannels = input.channels();

	//这是卷积结果，其通道数量因原始图像的通道数而定
	Mat DstImg;
	nchannels == 1 ? DstImg.create(rows, cols, CV_8UC1) :
		nchannels == 3 ? DstImg.create(rows, cols, CV_8UC3) : exit(0);

	//结果的指针
	unsigned char* pDst = DstImg.data;
	int p1, p2, p3, p4, p5, p6, p7, p8, p9;
	int record;

	for (int i = 0; i < nchannels; i++) {
		for (int j = 1; j < rows - 1; j++) {//第一行和最后一行不作为模板中心处理
			for (int k = 1; k < cols - 1; k++) {
				p1 = kernel[0] * pSrc[((j - 1) * cols + (k - 1)) * nchannels + i];
				p2 = kernel[1] * pSrc[((j - 1) * cols + k) * nchannels + i];
				p3 = kernel[2] * pSrc[((j - 1) * cols + (k + 1)) * nchannels + i];
				p4 = kernel[3] * pSrc[(j * cols + (k - 1)) * nchannels + i];
				p5 = kernel[4] * pSrc[(j * cols + k) * nchannels + i];
				p6 = kernel[5] * pSrc[(j * cols + (k + 1)) * nchannels + i];
				p7 = kernel[6] * pSrc[((j + 1) * cols + (k - 1)) * nchannels + i];
				p8 = kernel[7] * pSrc[((j + 1) * cols + k) * nchannels + i];
				p9 = kernel[8] * pSrc[((j + 1) * cols + (k + 1)) * nchannels + i];

				record = p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
				if (record > 255)
					pDst[(j * cols + k) * nchannels + i] = 255;
				else if (record < 0)
					pDst[(j * cols + k) * nchannels + i] = 0;
				else
					pDst[(j * cols + k) * nchannels + i] = record;
			}
		}
	}

	return DstImg;
}

Mat ImageFactory::Median(const Mat input)
{
	//获得原始图像的内容和参数
	unsigned char* pSrc = input.data;
	int rows = input.rows;
	int cols = input.cols;
	int nchannels = input.channels();

	//这是卷积结果，其通道数量因原始图像的通道数而定
	Mat DstImg;
	nchannels == 1 ? DstImg.create(rows, cols, CV_8UC1) :
		nchannels == 3 ? DstImg.create(rows, cols, CV_8UC3) : exit(0);

	//结果的指针
	unsigned char* pDst = DstImg.data;
	int pixels[9] = {};
	int record = 0;


	for (int i = 0; i < nchannels; i++) {
		for (int j = 1; j < rows - 1; j++) {//第一行和最后一行不作为模板中心处理
			for (int k = 1; k < cols - 1; k++) {
				pixels[0] = pSrc[((j - 1) * cols + (k - 1)) * nchannels + i];
				pixels[1] = pSrc[((j - 1) * cols + k) * nchannels + i];
				pixels[2] = pSrc[((j - 1) * cols + (k + 1)) * nchannels + i];
				pixels[3] = pSrc[(j * cols + (k - 1)) * nchannels + i];
				pixels[4] = pSrc[(j * cols + k) * nchannels + i];
				pixels[5] = pSrc[(j * cols + (k + 1)) * nchannels + i];
				pixels[6] = pSrc[((j + 1) * cols + (k - 1)) * nchannels + i];
				pixels[7] = pSrc[((j + 1) * cols + k) * nchannels + i];
				pixels[8] = pSrc[((j + 1) * cols + (k + 1)) * nchannels + i];

				sort(pixels, pixels + 8);
				record = pixels[5];

				if (record > 255)
					pDst[(j * cols + k) * nchannels + i] = 255;
				else if (record < 0)
					pDst[(j * cols + k) * nchannels + i] = 0;
				else
					pDst[(j * cols + k) * nchannels + i] = record;
			}
		}
	}

	return DstImg;
}

Mat ImageFactory::OtsuThresh(const Mat SrcImg)
{
	Mat DstImg;
	int rows = SrcImg.rows;
	int cols = SrcImg.cols;

	int pixelCount[256] = { 0 };//存储对应灰度像素的个数，灰度值和索引相对应
	int threshold = 0;//threshold为阈值

	DstImg.create(rows, cols, CV_8UC1);

	uchar* pSrc = SrcImg.data;
	uchar* pDst = DstImg.data;

	//检索每一个灰度值，统计个数
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			pixelCount[(int)pSrc[i * cols + j]]++;
		}
	}

	double g0 = 0;

	for (int t = 0; t < 256; t++) {
		double w0 = 0, w1 = 0;
		double u0 = 1, u1 = 0;
		for (int i = 0; i < t; i++) {
			w0 += pixelCount[i];//背景频率
			u0 += i * pixelCount[i];//背景平均灰度
		}
		if (w0 < 10e-30)u0 = 0;
		else u0 /= w0;
		w0 /= (rows * cols);

		for (int j = t; j < 256; j++) {
			w1 += pixelCount[j];//前景频率
			u1 += j * pixelCount[j];//前景平均灰度
		}
		if (w1 < 10e-30)u1 = 0;
		else u1 /= w1;
		w1 /= (rows * cols);

		double g1 = w0 * w1 * (u1 - u0) * (u1 - u0);//类间方差

		if (g1 > g0) {
			g0 = g1;
			threshold = t;//t为使最大类间方差的值
		}
	}

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (pSrc[i * cols + j] > threshold)
				pDst[i * cols + j] = 255;//前景
			else
				pDst[i * cols + j] = 0;//背景
		}
	}

	return DstImg;
}

Mat ImageFactory::StateThresh(const Mat SrcImg)
{
	Mat DstImg;

	int rows = SrcImg.rows;
	int cols = SrcImg.cols;

	int pixelCount[256] = { 0 };
	int threshold = 0;

	DstImg.create(rows, cols, CV_8UC1);

	uchar* pSrc = SrcImg.data;
	uchar* pDst = DstImg.data;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			pixelCount[(int)pSrc[i * cols + j]]++;
		}
	}

	int newthreshold;
	newthreshold = (0 + 255) / 2;
	int IterationTimes;
	for (IterationTimes = 0; threshold != newthreshold && IterationTimes < 100; IterationTimes++) {
		threshold = newthreshold;
		double w0 = 0, w1 = 0;
		double u0 = 1, u1 = 0;


		for (int i = 0; i < threshold; i++) {
			w0 += pixelCount[i];
			u0 += i * pixelCount[i];
		}
		if (w0 < 10e-30)u0 = 0;
		else u0 /= w0;
		w0 /= (rows * cols);

		for (int j = threshold + 1; j < 256; j++) {
			w1 += pixelCount[j];
			u1 += j * pixelCount[j];
		}
		if (w1 < 10e-30)u1 = 0;
		else u1 /= w1;
		w1 /= (rows * cols);

		newthreshold = int((u1 + u0) / 2);
	}

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (pSrc[i * cols + j] > threshold)
				pDst[i * cols + j] = 255;
			else
				pDst[i * cols + j] = 0;
		}
	}

	return DstImg;
}

