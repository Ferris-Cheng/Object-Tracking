//将手机拍摄得到的YUV图像，读取，选择，并保存到各个类中，jpg
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <string>
#include <numeric>

#include<io.h>
#include<time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include"newTracker.h"
using namespace std;
using namespace cv;
std::string tName(const int i) {
	if (i < 0) {
		return "error";
	}
	else if (i < 10) {
		return "img0000" + to_string(i);
	}
	else if (i < 100) {
		return "img000" + to_string(i);
	}
	else if (i < 1000) {
		return "img00" + to_string(i);
	}
	else if (i < 10000) {
		return "img0" + to_string(i);
	}
	else {
		return "img" + to_string(i);
	}
}
int main(int argc, char* argv[]) {

	bool hog = false;
	bool lab = false;
	//bool cn = false;

	bool colorhist = true;

	bool fixedwindow = true;
	bool multiscale = true;

	//create
	newTracker tracker(hog, lab, colorhist, fixedwindow, multiscale);

	const char *imgAll = "D:\\2-resources\\2.2-ObjcetTracking\\Test_case\\20180806cheng\\*.yuv";
	struct _finddata_t fileinfo;
	long handle;
	std::string tempPath = "D:\\2-resources\\2.2-ObjcetTracking\\Test_case\\20180806cheng\\";
	std::string tempName;
	std::string temp;
	std::string savePath;
	std::string saveTempName;
	int i1, i2, i3, i4, i5;
	cv::Mat image;
	cv::Size imgYUVSize;
	imgYUVSize.width = 768;
	imgYUVSize.height = 576;
	//handle = _findfirst(imgAll, &fileinfo);
	for (int i = 0; i < 6735; ++i) {
		cout << i << endl;
		if (i == 0) {
			i1 = 0;
			i2 = 0;
			i3 = 0;
			i4 = 0;
			i5 = 0;
			handle= _findfirst(imgAll, &fileinfo);
			if (handle == -1)
				cout << "fail open yuv image";
			else {
				tempName = tempPath + fileinfo.name;
				image = tracker.readYUVImage(tempName, imgYUVSize);
			}
		}
		else {
			if (!_findnext(handle, &fileinfo)) {
				temp = fileinfo.name;
				if (fileinfo.name[temp.size()-6]=='0') {
					tempName = tempPath + fileinfo.name;
					image = tracker.readYUVImage(tempName, imgYUVSize);
					transpose(image, image);
					flip(image, image, 1);
					if (i >= 3110 && i <= 3488) {      //car
						savePath = "D:\\2-resources\\2.2-ObjcetTracking\\Test_case\\0806car\\";
						saveTempName = savePath + tName(i1) + ".jpg";
						i1++;
						imwrite(saveTempName, image);
					}
					if (i >= 3505 && i <= 4270) {       //flowers1
						savePath = "D:\\2-resources\\2.2-ObjcetTracking\\Test_case\\0806flower1\\";
						saveTempName = savePath + tName(i2) + ".jpg";
						i2++;
						imwrite(saveTempName, image);
					}
					if (i >= 4478 && i <= 4831) {       //moblie
						savePath = "D:\\2-resources\\2.2-ObjcetTracking\\Test_case\\0806batterycar\\";
						saveTempName = savePath + tName(i3) + ".jpg";
						i3++;
						imwrite(saveTempName, image);
					}
					if (i >= 5157 && i <= 6130) {		//flowers2
						savePath = "D:\\2-resources\\2.2-ObjcetTracking\\Test_case\\0806flower2\\";
						saveTempName = savePath + tName(i4) + ".jpg";
						i4++;
						imwrite(saveTempName, image);
					}
					if (i >= 6480 && i <= 6691) {		//people
						savePath = "D:\\2-resources\\2.2-ObjcetTracking\\Test_case\\0806walkingpeople\\";
						saveTempName = savePath + tName(i5) + ".jpg";
						i5++;
						imwrite(saveTempName, image);
					}
				}
			}
		}
	}
	_findclose(handle);
	return 0;
}
