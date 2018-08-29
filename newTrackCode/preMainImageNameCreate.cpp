//为了方便程序读取，创建images.txt文件，其包含了待测图像的名称
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
int main(int argc, char* argv[]) {
	const char *imgAll = "D:\\2-resources\\2.2-ObjcetTracking\\OTB\\Skater\\*.jpg";
	struct _finddata_t fileinfo;
	string t1 = "", t2 = "";
	long handle;
	ofstream  f("D:\\2-resources\\2.2-ObjcetTracking\\OTB\\Skater\\images.txt", ios::trunc);
	if (!f.is_open()) {
		cout << "error!" << endl;
	}
	for (int i = 0; i < 1500; ++i) {
		if (i == 0) {
			handle = _findfirst(imgAll, &fileinfo);
		}
		else if (!_findnext(handle, &fileinfo)) {
			;
		}
		else {
			;
		}
		t2 = t1;
		t1 = fileinfo.name;
		if (t1 != t2)
			f << fileinfo.name;
		else
			break;
		f << endl;
	}
	f.close();
}
