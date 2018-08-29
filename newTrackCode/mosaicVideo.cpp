//将多个视频拼接在一起
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <string>
#include <numeric>
#include <core/core.hpp>
#include<io.h>
#include<direct.h>
#include<opencv2\opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

vector<cv::Mat> read_images_in_folder(cv::String pattern) {
	vector<cv::String> fn;
	glob(pattern, fn, false);

	vector<cv::Mat> image;
	size_t num = fn.size();
	for (int i = 0; i < num; ++i)
		image.push_back(imread(fn[i]));

	return image;
}
int main() {

	std::string Path = "D:\\1-documents\\1.1-ObjectTracking\\1.1.4-CaseTest\\TestResults\\";
	std::string img1 = "newTrackerVideo821-M-H",
		img2 = "newTrackerVideo821-M-C",
		img3 = "newTrackerTest0806flower2814-M-H-L",
		img4 = "newTrackerVideo821-M-H-C";

	vector<cv::Mat> image1, image2, image3, image4;
	image1 = read_images_in_folder(Path + img1 + "\\" + "*.jpg");
	image2 = read_images_in_folder(Path + img2 + "\\" + "*.jpg");
	//image3 = read_images_in_folder(Path + img3 + "\\" + "*.jpg");
	image4 = read_images_in_folder(Path + img4 + "\\" + "*.jpg");
	if (image1.size() != image2.size()) {
		cout << "input error!" << endl;
	}
	else {
		vector<cv::Mat> output(image1.size());
		cv::Size tSize;
		tSize.width = 4;
		tSize.height = image1[0].rows;
		cv::Mat tMat(tSize, CV_8UC3, cv::Scalar(1)), t1;

		cv::Size imgSize;
		imgSize.width = image1[0].cols * 3 + tSize.width * 2;
		imgSize.height = image1[0].rows;
		//imgSize.width *= 0.3;
		//imgSize.height *= 0.3;
		cv::VideoWriter video(Path+"newTrackerVedio2"+".avi", CV_FOURCC('M', 'J', 'P', 'G'), 12.0, imgSize, 1);

		for (int i = 0; i < 300; ++i) {
			cout << i << endl;
			hconcat(image1[i],tMat, t1);
			hconcat(t1, image2[i], t1);
			hconcat(t1, tMat, t1);
			//hconcat(t1, image3[i], t1);
			//hconcat(t1, tMat, t1);
			hconcat(t1, image4[i], t1);

			resize(t1, t1, imgSize);
			video << t1;
		}
	}
	return 0;
}
