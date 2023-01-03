#include "ImageFactory.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


ImageFactory::ImageFactory()
{}

ImageFactory ImageFactory::ReadImage(const char* path, const ImreadModes modes)
{
	Mat output = imread(path, modes);

	// 判断文件是否正常打开  
	if (output.empty())
	{
		fprintf(stderr, "Can not load image %s\n", path);
		waitKey(5000);  // 等待6000 ms后窗口自动关闭   
		exit(0);
	}

	this->matrixCopy = output;

	return *this;
}

void ImageFactory::ShowImage()
{
	imshow("image", this->matrixCopy);
	waitKey(0);
}

ImageFactory ImageFactory::Filter(const String mode)
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
		this->matrixCopy = this->Convolution(this->matrixCopy, laplacian);
	}
	else if (mode == "低通滤波")
	{
		this->matrixCopy = this->Convolution(this->matrixCopy, average);
	}
	else if (mode == "中值滤波") {
		this->matrixCopy = this->Median(this->matrixCopy);
	}
	else {
		cerr << "滤波模式选择错误，没有这种滤波模式" << endl;
		exit(0);
	}

	return *this;
}

ImageFactory ImageFactory::GrayTrans(const double alpha, const double beta)
{
	//获取原图像的值和其他参数
	unsigned char* pSrc = this->matrixCopy.data;
	int rows = this->matrixCopy.rows;
	int cols = this->matrixCopy.cols;
	int nchannels = this->matrixCopy.channels();

	Mat DstImg;
	//如果不是 1 或者 3 通道就直接挂
	nchannels == 1 ? DstImg.create(rows, cols, CV_8UC1) :
		nchannels == 3 ? DstImg.create(rows, cols, CV_8UC3) : exit(0);

	unsigned char* pDst = DstImg.data;
	double record;

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

	this->matrixCopy = DstImg;

	return *this;
}

double ImageFactory::CompareHist(ImageFactory image1, ImageFactory image2)
{
	// 计算模板图像的直方图
	Mat hist1;
	int channels[] = { 0 };
	int histSize[] = { 256 };
	float range[] = { 0, 256 };
	const float* ranges[] = { range };
	calcHist(image1.getMatrix(), 1, channels, Mat(), hist1, 1, histSize, ranges, true, false);

	// 规格化直方图
	normalize(hist1, hist1, 0, 1, NORM_MINMAX, -1, Mat());

	// 计算待匹配图像的直方图
	Mat hist2;
	calcHist(image2.getMatrix(), 1, channels, Mat(), hist2, 1, histSize, ranges, true, false);

	// 规格化直方图
	normalize(hist2, hist2, 0, 1, NORM_MINMAX, -1, Mat());

	// 进行直方图匹配
	return compareHist(hist1, hist2, HISTCMP_CORREL);
}

ImageFactory ImageFactory::Binarization(const String mode)
{
	Mat gray;
	cvtColor(this->matrixCopy, gray, COLOR_BGR2GRAY);

	if (mode == "状态法") {
		this->matrixCopy = this->StateThresh(gray);
	}
	else if (mode == "判断分析法") {
		this->matrixCopy = this->OtsuThresh(gray);
	}
	else
	{
		cerr << "输入的二值化模式不存在" << endl;
		exit(0);
	}

	return *this;
}

ImageFactory ImageFactory::Translation(const double x, const double y)
{
	int rows = this->matrixCopy.rows;
	int cols = this->matrixCopy.cols;
	int nchannels = this->matrixCopy.channels();

	Mat DstImg;
	//输出图像初始化
	if (nchannels == 1) {
		DstImg = Mat::zeros(rows, cols, CV_8UC1);
	}
	else if (nchannels == 3) {
		DstImg = Mat::zeros(rows, cols, CV_8UC3);
	}

	uchar* pSrc = this->matrixCopy.data;
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
					pDst[(i * cols + j) * nchannels + k] = (1 - dv) * (1 - du) * pSrc[(top * this->matrixCopy.cols + left) * nchannels + k] + (1 - dv) * du * pSrc[(top * this->matrixCopy.cols + right) * nchannels + k] + dv * (1 - du) * pSrc[(bottom * this->matrixCopy.cols + left) * nchannels + k] + dv * du * pSrc[(bottom * this->matrixCopy.cols + right) * nchannels + k];
				}
			}

		}
	}

	this->matrixCopy = DstImg;

	return *this;
}

ImageFactory ImageFactory::TransScale(const double x, const double y)
{
	int rows = round(this->matrixCopy.rows * x);
	int cols = round(this->matrixCopy.cols * y);
	int nchannels = this->matrixCopy.channels();

	Mat DstImg;
	//输出图像初始化
	if (nchannels == 1) {
		DstImg = Mat::zeros(rows, cols, CV_8UC1);
	}
	else if (nchannels == 3) {
		DstImg = Mat::zeros(rows, cols, CV_8UC3);
	}

	uchar* pSrc = this->matrixCopy.data;
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
			if (u >= 0 && v >= 0 && u <= this->matrixCopy.cols - 1 && v <= this->matrixCopy.rows - 1) {
				int top = floor(v), bottom = ceil(v), left = floor(u), right = ceil(u); //与映射到原图坐标相邻的四个像素点的坐标
				double dv = v - top; //dv为坐标 行 的小数部分(坐标偏差)
				double du = u - left; //du为坐标 列 的小数部分(坐标偏差)

				for (int k = 0; k < nchannels; k++) {
					pDst[(i * cols + j) * nchannels + k] = (1 - dv) * (1 - du) * pSrc[(top * this->matrixCopy.cols + left) * nchannels + k] + (1 - dv) * du * pSrc[(top * this->matrixCopy.cols + right) * nchannels + k] + dv * (1 - du) * pSrc[(bottom * this->matrixCopy.cols + left) * nchannels + k] + dv * du * pSrc[(bottom * this->matrixCopy.cols + right) * nchannels + k];
				}

			}

		}
	}

	this->matrixCopy = DstImg;

	return *this;
}

ImageFactory ImageFactory::TransRotate(double theta)
{
	theta = theta * CV_PI / 180;
	int rows = round(fabs(this->matrixCopy.rows * cos(theta)) + fabs(this->matrixCopy.cols * sin(theta)));
	int cols = round(fabs(this->matrixCopy.cols * cos(theta)) + fabs(this->matrixCopy.rows * sin(theta)));
	int nchannels = this->matrixCopy.channels();

	Mat DstImg;
	//输出图像初始化
	if (nchannels == 1) {
		DstImg = Mat::zeros(rows, cols, CV_8UC1);
	}
	else if (nchannels == 3) {
		DstImg = Mat::zeros(rows, cols, CV_8UC3);
	}

	uchar* pSrc = this->matrixCopy.data;
	uchar* pDst = DstImg.data;

	//构造旋转变换矩阵
	Mat T1 = (Mat_<double>(3, 3) << 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, -0.5 * this->matrixCopy.cols, 0.5 * this->matrixCopy.rows, 1.0);
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
			if (u >= 0 && v >= 0 && u <= this->matrixCopy.cols - 1 && v <= this->matrixCopy.rows - 1) {
				int top = floor(v), bottom = ceil(v), left = floor(u), right = ceil(u); //与映射到原图坐标相邻的四个像素点的坐标
				double dv = v - top; //dv为坐标 行 的小数部分(坐标偏差)
				double du = u - left; //du为坐标 列 的小数部分(坐标偏差)

				for (int k = 0; k < nchannels; k++) {
					pDst[(i * cols + j) * nchannels + k] = (1 - dv) * (1 - du) * pSrc[(top * this->matrixCopy.cols + left) * nchannels + k] + (1 - dv) * du * pSrc[(top * this->matrixCopy.cols + right) * nchannels + k] + dv * (1 - du) * pSrc[(bottom * this->matrixCopy.cols + left) * nchannels + k] + dv * du * pSrc[(bottom * this->matrixCopy.cols + right) * nchannels + k];
				}
			}


		}
	}

	this->matrixCopy = DstImg;

	return *this;
}

Mat* ImageFactory::getMatrix(void)
{
	return &this->matrixCopy;
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
	double p1, p2, p3, p4, p5, p6, p7, p8, p9;
	double record;

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
	Mat img = SrcImg.clone();
	int height = SrcImg.rows; //number of rows
	int width = SrcImg.cols; //number of colums
	int T = 0;//阈值
	int countdata[256];//统计灰度值的数组
	memset(countdata, 0, sizeof(countdata));//数组初始化
	int w = 0; int w1 = 0; int w2 = 0;//像素总数，类1像素总数，类2像素总数
	double d1 = 0.0; double d2 = 0.0; double d3 = 0.0; double d4 = 0.0; double max = 0.0;//类1间方差，类2间方差，类内方差，类间方差
	double ip = 0.0; double ip1 = 0.0;//各灰度值与个数的乘积
	double ratio = 0.0;//方差比值
	int graymin = 255; int graymax = 0;
	//统计各个灰度值像素数
	for (int i = 0; i < height; i++)  //循环图像高度
	{
		for (int j = 0; j < width; j++)  //循环图像宽度
		{
			countdata[SrcImg.at<uchar>(i, j)]++;//统计灰度个数
			if (SrcImg.at<uchar>(i, j) > graymax)graymax = SrcImg.at<uchar>(i, j);
			if (SrcImg.at<uchar>(i, j) < graymin) graymin = SrcImg.at<uchar>(i, j);
			if (graymin == 0)  graymin++;
		}
	}
	//计算像素总数与总灰度值
	for (int k = graymin; k <= graymax; k++)
	{
		w += countdata[k];//像素总数
		ip += (double)k * (double)countdata[k];//像素值与像素数的乘积
	}
	//求阈值
	for (int k = graymin; k <= graymax; k++)
	{
		w1 += countdata[k];//类1像素总数
		if (!w1)
		{
			continue;
		}
		w2 = w - w1;//类2像素总数
		if (w2 == 0)
		{
			break;
		}
		ip1 += (double)k * countdata[k];
		//计算类1均值
		double  m1 = ip1 / w1;
		//计算类1间方差
		for (int n = graymin; n <= k; n++)
		{
			d1 += ((n - m1) * (n - m1) * countdata[n]);
		}
		//计算类2均值
		double m2 = (ip - ip1) / w2;
		//计算类2间方差
		for (int m = k + 1; m <= graymax; m++)
		{
			d2 += ((m - m2) * (m - m2) * countdata[m]);
		}

		//计算类内方差
		d3 = d1 * w1 + d2 * w2;
		//计算类间方差
		d4 = (double)w1 * (double)w2 * (m1 - m2) * (m1 - m2);
		//类内方差与类间方差比值
		if (d3 != 0)
			ratio = d4 / d3;
		if (ratio > max)
		{
			max = ratio;//找到比值最大值
			T = k; //找到比值最大值时T的值 
		}
	}
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (SrcImg.at<uchar>(i, j) > T)//利用阈值进行二值化
				img.at<uchar>(i, j) = 255;
			else
				img.at<uchar>(i, j) = 0;
		}
	}

	std::cout << "阈值为" << endl;
	std::cout << T << endl;

	return img;
}

Mat ImageFactory::StateThresh(const Mat SrcImg)
{
	Mat img = SrcImg.clone();////复制矩阵头，且复制一份新数据，克隆
	int height = SrcImg.rows; //number of rows
	int width = SrcImg.cols; //number of colums

	int countdata[256] = { 0 };
	int graymax = 0; int graymin = 255;
	for (int i = 0; i < height; i++)  //循环图像高度
	{
		for (int j = 0; j < width; j++)  //循环图像宽度
		{
			countdata[SrcImg.at<uchar>(i, j)]++;//统计灰度个数
			if (SrcImg.at<uchar>(i, j) > graymax)graymax = SrcImg.at<uchar>(i, j);//找到灰度的最大最小值
			if (SrcImg.at<uchar>(i, j) < graymin) graymin = SrcImg.at<uchar>(i, j);
		}
	}
	int peak1 = 0; int peak2 = 0;
	for (int i = 1; i <= 254; i++)//循环找出第一个波峰所对应的灰度值
	{
		if (countdata[i] > countdata[i - 1] && countdata[i] > countdata[i + 1])
		{
			peak1 = i;
		}
	}
	for (int j = 254; j >= 1; j--)//循环找出第二个波峰对应的灰度值
	{
		if (countdata[j] > countdata[j - 1] && countdata[j] > countdata[j + 1])
		{

			if (peak1 != j)
				peak2 = j;
		}
	}

	int valley = (peak1 + peak2) / 2;//找峰谷
	if (countdata[valley] > countdata[valley + 1])//如果两个波峰的平均值比右边值大，说明波谷在右边
	{
		for (int i = valley; i < peak2; i++) //波谷向右找
		{
			if (countdata[i + 1] > countdata[i])
				valley = i;
		}
	}
	if (countdata[valley] > countdata[valley - 1])
	{
		for (int i = valley; i > peak1; i--)//波谷向左找
		{
			if (countdata[i] > countdata[i - 1])
				valley = i;
		}
	}
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (SrcImg.at<uchar>(i, j) > valley)//二值化处理
				img.at<uchar>(i, j) = 255;
			else
				img.at<uchar>(i, j) = 0;
		}
	}
	cout << "阈值为" << endl;
	cout << valley << endl;

	return img;
}

