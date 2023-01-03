
#include <iostream>
#include "ImageFactory.h"

using namespace std;
using namespace cv;


int main()
{
	/*
		以下全是交互代码，实际的图像操作已经被封装到了ImageFactory类中

		交互主界面

		导入图像----->展示图像、图像滤波、灰度变换、直方图匹配、图像二值化、图像运动
	*/

	//设置基本实例
	ImageFactory instance;
	ImageFactory another;

	//此为选项指标，每次均观察其值
	String flag = "";

	//图像加载界面
	String path = "../testing/mir.jpg";
	cout << "请输入你要操作图像的 \033[32m绝对地址\033[0m ，如果输入 \033[32mdefault\033[0m ，将会使用默认图像" << endl << endl << ">> " << flush;
	cin >> flag;
	flag == "default" ? path : path = flag;
	instance = instance.ReadImage(path.c_str(), IMREAD_COLOR);
	cout << endl << "打开成功，图像路径是" << path << endl << endl;

	//过渡缓冲
	system("pause");
	system("cls");

	//主界面
	cout << "请选择要对此图像进行的操作，每次操作结束后均会展示图像，输入对应的阿拉伯数字即可\n" << endl << "\t①展示图像\t②图像滤波\t③灰度变换\t④直方图匹配\t⑤图像二值化\t⑥图像运动" << endl << endl << ">> " << flush;
	cin >> flag;
	system("cls");

	switch (flag[0] - '0')
	{
	case 1:
		instance.ShowImage();
		break;
	case 2:
		cout << "你选择了图像滤波，请选择滤波方式\n\n\t①高通滤波\t②低通滤波\t③中值滤波" << endl << endl << " >>" << flush;
		cin >> flag;
		if (flag == "1") {
			instance.Filter("高通滤波").ShowImage();
		}
		else if (flag == "2") {
			instance.Filter("低通滤波").ShowImage();
		}
		else if (flag == "3") {
			instance.Filter("中值滤波").ShowImage();
		}
		else {
			cerr << "模式选择错误" << endl;
			exit(0);
		}
		break;

	case 3:
		double k, b;
		cout << "你选择了灰度线性变换，其原理是将图像各个通道每个像素的灰度按照\033[32m y = kx + b \033[0m的形式实现映射，同时将灰度控制在量化位数所能接受的灰度级之内"
			<< endl << endl << "请输入 k 和 b 的值" << endl << endl << ">> " << flush;
		cin >> k >> b;
		instance.GrayTrans(k, b).ShowImage();
		break;

	case 4:
		cout << "直方图匹配需要另一张图片，请输入另一张图片的绝对路径" << endl << endl << ">> " << flush;
		cin >> flag;
		another = another.ReadImage(flag.c_str(), IMREAD_COLOR);
		cout << endl << "两张图片的匹配度为 \033[32m" << ImageFactory::CompareHist(instance, another) << "\033[0m" << endl;
		break;

	case 5:
		cout << "你选择了图像二值化，我们提供状态法和判断分析法，请选择\n\n\t①状态法\t\t②判断分析法"
			<< endl << endl << ">> " << flush;
		cin >> flag;
		flag == "1" ? instance.Binarization("状态法").ShowImage() :
			flag == "2" ? instance.Binarization("判断分析法").ShowImage() : exit(0);
		break;

	case 6:
		double moveX, moveY, scaleX, scaleY, theta;
		cout << "你选择了图像运动，下面是我们有三种形式的运动方式，分别是图像在 x y 轴上的平移，拉伸，旋转"
			<< endl << endl << "请输入依次输入图像的\033[32m moveX moveY scaleX scaleY theta \033[0m"
			<< "\n\n注意：\n\tmoveX 和 moveY 的单位是像素"
			<< "\n\tscaleX scaleY 是比例，一般 1.5 左右效果就相当显著"
			<< "\n\ttheta 是角度 ，范围是 [0,360]"
			<< endl << endl << ">> " << flush;
		cin >> moveX >> moveY >> scaleX >> scaleY >> theta;
		instance.Translation(moveX, moveY).TransScale(scaleX, scaleY).TransRotate(theta).ShowImage();
		break;

	default:
		cerr << "输入了错误的数字" << endl;
		exit(0);
		break;
	}

	//程序结束界面

	system("cls");
	cout << "计算完成，图像已关闭，程序退出" << endl;
}

