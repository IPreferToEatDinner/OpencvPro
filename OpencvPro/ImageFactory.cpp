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

	// �ж��ļ��Ƿ�������  
	if (output.empty())
	{
		fprintf(stderr, "Can not load image %s\n", path);
		waitKey(5000);  // �ȴ�6000 ms�󴰿��Զ��ر�   
		exit(0);
	}

	this->matrixCopy = output;

	return *this;
}

void ImageFactory::ShowImage()const
{
	imshow("Result", this->matrixCopy);
	waitKey(0);
	destroyAllWindows();
}

void ImageFactory::ShowWithoutClose()const
{
	imshow("Row", this->matrixCopy);
}

ImageFactory ImageFactory::Filter(const String mode, const double kernel[9])
{
	double laplacian[9] = {
		0, -1, 0,
		-1, 4, -1,
		0, -1, 0 };

	double average[9] = {
		1 / 9.0, 1 / 9.0, 1 / 9.0,
		1 / 9.0, 1 / 9.0, 1 / 9.0,
		1 / 9.0, 1 / 9.0, 1 / 9.0 };

	if (kernel != nullptr) {
		this->matrixCopy = this->Convolution(this->matrixCopy, kernel);
	}
	else if (mode == "��ͨ�˲�") {
		this->matrixCopy = this->Convolution(this->matrixCopy, laplacian);
	}
	else if (mode == "��ͨ�˲�")
	{
		this->matrixCopy = this->Convolution(this->matrixCopy, average);
	}
	else if (mode == "��ֵ�˲�") {
		this->matrixCopy = this->Median(this->matrixCopy);
	}
	else {
		cerr << "�˲�ģʽѡ�����û�������˲�ģʽ" << endl;
		exit(0);
	}

	return *this;
}

ImageFactory ImageFactory::GrayTrans(const double alpha, const double beta)
{
	//��ȡԭͼ���ֵ����������
	unsigned char* pSrc = this->matrixCopy.data;
	int rows = this->matrixCopy.rows;
	int cols = this->matrixCopy.cols;
	int nchannels = this->matrixCopy.channels();

	Mat DstImg;
	//������� 1 ���� 3 ͨ����ֱ�ӹ�
	nchannels == 1 ? DstImg.create(rows, cols, CV_8UC1) :
		nchannels == 3 ? DstImg.create(rows, cols, CV_8UC3) : exit(0);

	unsigned char* pDst = DstImg.data;
	double record;

	//ͨ��ѭ��
	for (int i = 0; i < nchannels; ++i)
	{
		//��ѭ��
		for (int j = 0; j < rows; ++j)
		{
			//��ѭ��
			for (int l = 0; l < cols; ++l)
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
	// ����ģ��ͼ���ֱ��ͼ
	Mat hist1;
	int channels[] = { 0 };
	int histSize[] = { 256 };
	float range[] = { 0, 256 };
	const float* ranges[] = { range };
	calcHist(image1.getMatrix(), 1, channels, Mat(), hist1, 1, histSize, ranges, true, false);

	// ���ֱ��ͼ
	normalize(hist1, hist1, 0, 1, NORM_MINMAX, -1, Mat());

	// �����ƥ��ͼ���ֱ��ͼ
	Mat hist2;
	calcHist(image2.getMatrix(), 1, channels, Mat(), hist2, 1, histSize, ranges, true, false);

	// ���ֱ��ͼ
	normalize(hist2, hist2, 0, 1, NORM_MINMAX, -1, Mat());

	// ����ֱ��ͼƥ��
	return compareHist(hist1, hist2, HISTCMP_CORREL);
}

ImageFactory ImageFactory::Binarization(const String mode)
{
	Mat gray;
	cvtColor(this->matrixCopy, gray, COLOR_BGR2GRAY);

	if (mode == "״̬��") {
		this->matrixCopy = this->StateThresh(gray);
	}
	else if (mode == "�жϷ�����") {
		this->matrixCopy = this->OtsuThresh(gray);
	}
	else
	{
		cerr << "����Ķ�ֵ��ģʽ������" << endl;
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
	//���ͼ���ʼ��
	nchannels == 1 ? DstImg.create(rows, cols, CV_8UC1) :
		nchannels == 3 ? DstImg.create(rows, cols, CV_8UC3) : exit(0);

	uchar* pSrc = this->matrixCopy.data;
	uchar* pDst = DstImg.data;

	//����ƽ�Ʊ任����
	Mat T = (Mat_<double>(3, 3) <<
		1, 0, 0,
		0, 1, 0,
		x, y, 1);

	Mat T_inv = T.inv();//����

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			//��i��j�е���������(j,i)
			Mat dst_xy = (Mat_<double>(1, 3) << j, i, 1);
			Mat src_uv = dst_xy * T_inv;

			//ԭͼ��ĺ�����-->ͼ�������
			double u = src_uv.at<double>(0, 0);

			//ԭͼ���������-->ͼ�������
			double v = src_uv.at<double>(0, 1);

			//˫���Բ�ֵ��
			if (u >= 0 && v >= 0 && u <= cols - 1 && v <= rows - 1) {//�ж϶�Ӧ��(u,v)�Ƿ���ԭͼ��Χ�ڣ����Ƿ�Խ��
				int top = floor(v), bottom = ceil(v), left = floor(u), right = ceil(u);
				double dv = v - top, du = u - left;//����ƫ��ֵ����С������

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

	//���ͼ���ʼ��
	nchannels == 1 ? DstImg.create(rows, cols, CV_8UC1) :
		nchannels == 3 ? DstImg.create(rows, cols, CV_8UC3) : exit(0);

	uchar* pSrc = this->matrixCopy.data;
	uchar* pDst = DstImg.data;

	//�������ű任����
	Mat T = (Mat_<double>(3, 3) << x, 0, 0, 0, y, 0, 0, 0, 1);
	Mat T_inv = T.inv();//����

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			//��i��j�е���������(j,i)
			Mat dst_xy = (Mat_<double>(1, 3) << j, i, 1);
			Mat src_uv = dst_xy * T_inv;

			//ԭͼ��ĺ����꣬��Ӧͼ�������
			double u = src_uv.at<double>(0, 0);
			//ԭͼ��������꣬��Ӧͼ�������
			double v = src_uv.at<double>(0, 1);

			//˫���Բ�ֵ��
			if (u >= 0 && v >= 0 && u <= this->matrixCopy.cols - 1 && v <= this->matrixCopy.rows - 1) {
				int top = floor(v), bottom = ceil(v), left = floor(u), right = ceil(u); //��ӳ�䵽ԭͼ�������ڵ��ĸ����ص������
				double dv = v - top; //dvΪ���� �� ��С������(����ƫ��)
				double du = u - left; //duΪ���� �� ��С������(����ƫ��)

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
	//���ͼ���ʼ��
	nchannels == 1 ? DstImg.create(rows, cols, CV_8UC1) :
		nchannels == 3 ? DstImg.create(rows, cols, CV_8UC3) : exit(0);

	uchar* pSrc = this->matrixCopy.data;
	uchar* pDst = DstImg.data;

	//������ת�任����
	Mat T1 = (Mat_<double>(3, 3) <<
		1.0, 0.0, 0.0,
		0.0, -1.0, 0.0,
		-0.5 * this->matrixCopy.cols, 0.5 * this->matrixCopy.rows, 1.0);

	Mat T2 = (Mat_<double>(3, 3) <<
		cos(theta), -sin(theta), 0.0,
		sin(theta), cos(theta), 0.0,
		0.0, 0.0, 1.0);

	double t3[3][3] = {
		{ 1.0, 0.0, 0.0 },
		{ 0.0, -1.0, 0.0 },
		{ 0.5 * DstImg.cols, 0.5 * DstImg.rows ,1.0 }
	}; // ����ѧ�ѿ�������ӳ�䵽��ת���ͼ������


	Mat T3 = Mat(3.0, 3.0, CV_64FC1, t3);

	Mat T = T1 * T2 * T3;
	Mat T_inv = T.inv();//����

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			Mat dst_xy = (Mat_<double>(1, 3) << j, i, 1);//��i��j�е���������(j,i)
			Mat src_uv = dst_xy * T_inv;

			double u = src_uv.at<double>(0, 0);//ԭͼ��ĺ����꣬��Ӧͼ�������
			double v = src_uv.at<double>(0, 1);//ԭͼ��������꣬��Ӧͼ�������

			//˫���Բ�ֵ��
			if (u >= 0 && v >= 0 && u <= this->matrixCopy.cols - 1 && v <= this->matrixCopy.rows - 1) {
				int top = floor(v), bottom = ceil(v), left = floor(u), right = ceil(u); //��ӳ�䵽ԭͼ�������ڵ��ĸ����ص������
				double dv = v - top; //dvΪ���� �� ��С������(����ƫ��)
				double du = u - left; //duΪ���� �� ��С������(����ƫ��)

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

ImageFactory ImageFactory::ColorBalance(void)
{
	Mat img = this->matrixCopy.clone();
	int height = this->matrixCopy.rows;
	int width = this->matrixCopy.cols;
	double* Y = new double[height * width];//��ͼƬ������Ϣ
	double Ya = 0;
	int i, j = 0; double Ymax = 0;

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			Y[i * width + j] = 0.299 * this->matrixCopy.at<Vec3b>(i, j)[2] + 0.587 * this->matrixCopy.at<Vec3b>(i, j)[1] + 0.114 * this->matrixCopy.at<Vec3b>(i, j)[0];//ͼ�����ȷ���
			Ya += Y[i * width + j];
			if (Ymax < Y[i * width + j])//ͼ��������Ϣ
			{
				Ymax = Y[i * width + j];
			}
		}
	}

	Ya = Ya / (height * width);//��ȡͼ���ƽ������
	double Ra = 0, Ga = 0, Ba = 0;

	int num = 0;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (Y[i * width + j] < (0.95 * Ymax))
			{
				Ra += this->matrixCopy.at<Vec3b>(i, j)[2];
				Ga += this->matrixCopy.at<Vec3b>(i, j)[1];
				Ba += this->matrixCopy.at<Vec3b>(i, j)[0];
				num++;
			}
		}
	}

	//ƽ��ֵ��
	Ra /= num;
	Ga /= num;
	Ba /= num;

	//��ɫ����ϵ��
	double K[3];

	//��������ɫ����ƽ��ֵ���ֵ
	double maxB = 0;
	maxB = max(Ra, max(Ga, Ba));
	K[0] = maxB / Ba;//RGB��ɫ����ϵ��
	K[1] = maxB / Ga;
	K[2] = maxB / Ra;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				int temp = K[k] * this->matrixCopy.at<Vec3b>(i, j)[k];//ƽ����ֵ����ƽ��ǰ��ֵ����ϵ��
				if (temp < 0)
				{//�ж����ֵ�Ƿ��ڷ�Χ��
					img.at<Vec3b>(i, j)[k] = 0;
				}
				else if (temp > 255)
				{
					img.at<Vec3b>(i, j)[k] = 255;
				}
				else
					img.at<Vec3b>(i, j)[k] = temp;
			}
		}
	}

	this->matrixCopy = img;

	return *this;
}

Mat ImageFactory::Convolution(const Mat input, const double* kernel)
{
	//���ԭʼͼ������ݺͲ���
	unsigned char* pSrc = input.data;
	int rows = input.rows;
	int cols = input.cols;
	int nchannels = input.channels();

	//���Ǿ���������ͨ��������ԭʼͼ���ͨ��������
	Mat DstImg;
	nchannels == 1 ? DstImg.create(rows, cols, CV_8UC1) :
		nchannels == 3 ? DstImg.create(rows, cols, CV_8UC3) : exit(0);

	//�����ָ��
	unsigned char* pDst = DstImg.data;
	double p1, p2, p3, p4, p5, p6, p7, p8, p9;
	double record;

	for (int i = 0; i < nchannels; i++) {
		for (int j = 1; j < rows - 1; j++) {//��һ�к����һ�в���Ϊģ�����Ĵ���
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
	//���ԭʼͼ������ݺͲ���
	unsigned char* pSrc = input.data;
	int rows = input.rows;
	int cols = input.cols;
	int nchannels = input.channels();

	//���Ǿ���������ͨ��������ԭʼͼ���ͨ��������
	Mat DstImg;
	nchannels == 1 ? DstImg.create(rows, cols, CV_8UC1) :
		nchannels == 3 ? DstImg.create(rows, cols, CV_8UC3) : exit(0);

	//�����ָ��
	unsigned char* pDst = DstImg.data;
	int pixels[9] = {};
	int record = 0;


	for (int i = 0; i < nchannels; i++) {
		for (int j = 1; j < rows - 1; j++) {//��һ�к����һ�в���Ϊģ�����Ĵ���
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

	int T = 0;//��ֵ
	int countdata[256] = { 0 };//ͳ�ƻҶ�ֵ������


	double ip = 0.0, ip1 = 0.0;//���Ҷ�ֵ������ĳ˻�
	double ratio = 0.0;//�����ֵ
	int graymin = 255, graymax = 0;

	//ͳ�Ƹ����Ҷ�ֵ������
	for (int i = 0; i < height; i++)  //ѭ��ͼ��߶�
	{
		for (int j = 0; j < width; j++)  //ѭ��ͼ����
		{
			countdata[SrcImg.at<uchar>(i, j)]++;//ͳ�ƻҶȸ���
			if (SrcImg.at<uchar>(i, j) > graymax) {
				graymax = SrcImg.at<uchar>(i, j);
			}
			if (SrcImg.at<uchar>(i, j) < graymin) {
				graymin = SrcImg.at<uchar>(i, j);
			}
			if (graymin == 0) {
				graymin++;
			}
		}
	}

	//����������ǰ����������������������
	int w = 0, w1 = 0, w2 = 0;
	//���������������ܻҶ�ֵ
	for (int k = graymin; k <= graymax; k++)
	{
		w += countdata[k];//��������
		ip += (double)k * (double)countdata[k];//����ֵ���������ĳ˻�
	}

	//ǰ���䷽��󾰼䷽����ڷ����䷽��
	double d1 = 0.0, d2 = 0.0, d3 = 0.0, d4 = 0.0, max = 0.0;

	//����ֵ
	for (int k = graymin; k <= graymax; k++)
	{
		//ǰ����������
		w1 += countdata[k];
		if (!w1) {
			continue;
		}

		//����������
		w2 = w - w1;
		if (w2 == 0) {
			break;
		}
		ip1 += (double)k * countdata[k];

		//����ǰ����ֵ
		double  m1 = ip1 / w1;

		//����ǰ���䷽��
		for (int n = graymin; n <= k; n++) {
			d1 += ((n - m1) * (n - m1) * countdata[n]);
		}

		//����󾰾�ֵ
		double m2 = (ip - ip1) / w2;

		//����󾰼䷽��
		for (int m = k + 1; m <= graymax; m++)
		{
			d2 += ((m - m2) * (m - m2) * countdata[m]);
		}

		//�������ڷ���
		d3 = d1 * w1 + d2 * w2;

		//������䷽��
		d4 = (double)w1 * (double)w2 * (m1 - m2) * (m1 - m2);

		//���ڷ�������䷽���ֵ
		if (d3 != 0)
			ratio = d4 / d3;
		if (ratio > max) {
			max = ratio;//�ҵ���ֵ���ֵ
			T = k; //�ҵ���ֵ���ֵʱT��ֵ 
		}
	}

	//������ֵ���ж�ֵ��
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (SrcImg.at<uchar>(i, j) > T) {
				img.at<uchar>(i, j) = 255;
			}
			else {
				img.at<uchar>(i, j) = 0;
			}
		}
	}

	cout << endl << "��ֵΪ\033[32m" << T << "\033[0m" << endl;

	return img;
}

Mat ImageFactory::StateThresh(const Mat SrcImg)
{
	Mat img = SrcImg.clone();////���ƾ���ͷ���Ҹ���һ�������ݣ���¡
	int height = SrcImg.rows;
	int width = SrcImg.cols;

	int countdata[256] = { 0 };
	int graymax = 0; int graymin = 255;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			//ͳ�ƻҶȸ���
			countdata[SrcImg.at<uchar>(i, j)]++;
			if (SrcImg.at<uchar>(i, j) > graymax) { graymax = SrcImg.at<uchar>(i, j); }
			if (SrcImg.at<uchar>(i, j) < graymin) { graymin = SrcImg.at<uchar>(i, j); }
		}
	}

	int peak1 = 0; int peak2 = 0;

	//ѭ���ҳ���һ����������Ӧ�ĻҶ�ֵ
	for (int i = 1; i <= 254; i++)
	{
		if (countdata[i] > countdata[i - 1] && countdata[i] > countdata[i + 1]) {
			peak1 = i;
		}
	}

	//ѭ���ҳ��ڶ��������Ӧ�ĻҶ�ֵ
	for (int j = 254; j >= 1; j--) {
		if (countdata[j] > countdata[j - 1] && countdata[j] > countdata[j + 1])
		{
			if (peak1 != j)
				peak2 = j;
		}
	}
	//�ҷ��
	int valley = (peak1 + peak2) / 2;

	//������������ƽ��ֵ���ұ�ֵ��˵���������ұ�
	if (countdata[valley] > countdata[valley + 1]) {
		//����������
		for (int i = valley; i < peak2; i++)
		{
			if (countdata[i + 1] > countdata[i])
				valley = i;
		}
	}

	if (countdata[valley] > countdata[valley - 1]) {
		//����������
		for (int i = valley; i > peak1; i--)
		{
			if (countdata[i] > countdata[i - 1])
				valley = i;
		}
	}
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			//��ֵ������
			if (SrcImg.at<uchar>(i, j) > valley) {
				img.at<uchar>(i, j) = 255;
			}
			else {
				img.at<uchar>(i, j) = 0;
			}
		}
	}

	cout << endl << "��ֵΪ\033[32m" << valley << "\033[0m" << endl;

	return img;
}