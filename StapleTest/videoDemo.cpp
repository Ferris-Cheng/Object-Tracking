#include "D:\1-documents\\1.1-ObjectTracking\\1.1.4-CaseTest\\StapleTest\\temp\\staple_tracker.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <numeric>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//
#include<io.h>
#include<direct.h>
#include<time.h>
using namespace std;
using namespace cv;

int main(int argc, char * argv[]) {
	bool isstorage = false;
	bool isSaveVideo = false;
	bool show_visualization = true;
	//
	STAPLE_TRACKER staple;

	std::string imgSavePathTemp = "D:\\1-documents\\1.1-ObjectTracking\\1.1.4-CaseTest\\TestResults\\";
	std::string methodName = "stapleVideo";
	std::string imgSavePath = imgSavePathTemp + methodName;

	std::string savePathFiles;
	if (1) {
		savePathFiles = staple.saveImage(imgSavePath);
	}

	VideoCapture cap("D:\\2-resources\\2.2-ObjcetTracking\\Test_case\\screen-recording_20180723-151722.mp4");
	cv::Rect_<float> location;
	location.x = 780;
	location.y = 500;
	location.width = 200;
	location.height = 200;
	std::string savePath;
	Mat image;
	for (int i = 0; i < 500; i++) {
		cap >> image;

		if (i == 0) {
			staple.tracker_staple_initialize(image, location);
			staple.tracker_staple_train(image, true);
		}
		else {
			location = staple.tracker_staple_update(image);
			staple.tracker_staple_train(image, false);
		}

		if (show_visualization) {
			cv::putText(image, std::to_string(i + 1), cv::Point(20, 40), 6, 1,
				cv::Scalar(0, 255, 255), 2);
			//cv::rectangle(image, groundtruth_rect[frame], cv::Scalar(0, 255, 0), 2);
			cv::rectangle(image, location, cv::Scalar(0, 128, 255), 2);

			cv::imshow("STAPLEVideo", image);
			if (isstorage) {
				savePath = savePathFiles + "\\" + to_string(i) + ".jpg";
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
		if (isstorage) {
			//void saveVideo(string saveImgPath, string saveVideoPath, string videoNameTemp, cv::Size imgSize, bool hog, bool colorhist);
			staple.saveVideo(savePathFiles, imgSavePathTemp, methodName, imgSize);
		}
		else {
			cout << "Error: not save image";
		}
	}
	cv::destroyAllWindows();

	return 0;
}
