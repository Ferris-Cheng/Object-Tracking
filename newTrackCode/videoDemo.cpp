//读取视频文件，进行测试
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <string>
#include <numeric>

#include<time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include"newTracker.h"
using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {

	bool hog = true;
	bool lab = false;
	//bool cn = false;

	bool colorhist = false;

	bool fixedwindow = true;
	bool multiscale = true;

	bool isSaveImage = true;
	bool isSaveVideo = false;
	bool show_visualization = true;
	std::string savePath;
	std::string imgSavePathTemp = "D:\\1-documents\\1.1-ObjectTracking\\1.1.4-CaseTest\\TestResults\\";
	std::string methodName = "newTrackerVideo";
	std::string imgSavePath = imgSavePathTemp + methodName;

	std::string savePathFiles;
	
	//create
	newTracker tracker(hog, lab, colorhist, fixedwindow, multiscale);

	//frame
	Mat image;

	// Tracker results
	//Rect result;

	if (isSaveImage) {
		savePathFiles = tracker.saveImage(imgSavePath, hog, lab, colorhist, multiscale);
	}

	VideoCapture cap("D:\\2-resources\\2.2-ObjcetTracking\\Test_case\\screen-recording_20180814-152647.mp4");
	cv::Rect_<float> location;
	location.x = 380;
	location.y = 610;
	location.width = 450;
	location.height = 480;
	cv::Size imgSize1;
	imgSize1.width = (int)1080 * 0.3;
	imgSize1.height = (int)2044 * 0.3;
	for (unsigned int i = 0; i < 300; i++) {
		cap >> image;
		
		if (i == 0)
			tracker.init(location, image);
		else
			location = tracker.update(image);
		if (show_visualization) {
			cv::putText(image, "Hog"+std::to_string(i + 1), cv::Point(20, 40), 2, 1,cv::Scalar(0, 255, 255), 4);
			//cv::rectangle(image, groundtruth_rect[frame], cv::Scalar(0, 255, 0), 2);
			cv::rectangle(image, Point(location.x, location.y), Point(location.x + location.width, location.y + location.height), cv::Scalar(0, 255, 0), 8);
			cv::imshow("newtracker", image);

			//保存结果
			if (isSaveImage) {
				savePath = savePathFiles + "\\" + to_string(i) + ".jpg";
				resize(image, image, imgSize1);
				imwrite(savePath, image);
			}

			char key = cv::waitKey(1);
			if (key == 27 || key == 'q' || key == 'Q')
				break;
		}
	}
	cv::Size imgSize;
	imgSize.height = image.rows;
	imgSize.width = image.cols;
	if (isSaveVideo) {
		if (isSaveImage) {
			//saveVideo(string saveImgPath, string saveVideoPath, string videoNameTemp, cv::Size imgSize)
			tracker.saveVideo(savePathFiles, imgSavePathTemp, methodName, imgSize, multiscale, hog, lab, colorhist);
		}
		else {
			cout << "Error: not save image";
		}
	}
	cv::destroyAllWindows();
	return 0;
}
