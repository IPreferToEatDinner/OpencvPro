
#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "ImageFactory.h"

using namespace std;
using namespace cv;

int main()
{
	ImageFactory instance;

	char imageNameA[] = "C:\\Users\\seven_three\\Desktop\\实习\\testimg\\mir.jpg";
	char imageNameB[] = "C:\\Users\\seven_three\\Desktop\\实习\\testimg\\nir.jpg";

	// 读入图片
	Mat A = instance.ReadImage(imageNameA, IMREAD_COLOR);
	Mat B = instance.ReadImage(imageNameB, IMREAD_COLOR);

	//展示图片
	instance.ShowImage(A);

	//图像二值化
	//instance.ShowImage(instance.Binarization(A, "状态法"));
	//instance.ShowImage(instance.Binarization(A, "判断分析法"));

	//灰度变换
	//instance.ShowImage(instance.GrayTrans(A, 0.2, 1));

	//滤波
	//instance.ShowImage(instance.Filter(A, "中值滤波"));

	//直方图匹配
	//cout << instance.CompareHist(A, B) << endl;

	//图像平移
	//instance.ShowImage(instance.Translation(A, 100, 100));

	//图像缩放
	//instance.ShowImage(instance.TransScale(A, 1.3, 1.5));

	//图像旋转
	//instance.ShowImage(instance.TransRotate(A, 45));

	return 0;
}

