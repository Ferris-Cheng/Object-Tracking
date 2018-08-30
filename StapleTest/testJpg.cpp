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
#include "d:\\1-documents\\1.1-ObjectTracking\\1.1.4-CaseTest\\StapleTest\\temp\\staple_tracker.hpp"
using namespace std;
using namespace cv;
int main(int argc, char * argv[]) {
	STAPLE_TRACKER staple;

	//数据集路径
	std::string PATH_IMG_TOPCV = "D:\\2-resources\\2.2-ObjcetTracking\\OTB";
	std::string seq_name = "Girl2";
	std::string imgPath = PATH_IMG_TOPCV + "\\" + seq_name + "\\";

	std::string imgSavePathTemp = "D:\\1-documents\\1.1-ObjectTracking\\1.1.4-CaseTest\\TestResults\\";
	std::string methodName = "staple";
	std::string imgSavePath = imgSavePathTemp + methodName;

	//获取第一帧的region
	Rect result; // Tracker results
	ifstream groundtruthFile;
	string groundtruth = imgPath + "groundtruth.txt";
	groundtruthFile.open(groundtruth);
	string firstLine;
	getline(groundtruthFile, firstLine);
	groundtruthFile.close();
	istringstream ss(firstLine);
	char ch;
	ss >> result.x;
	ss >> ch;
	ss >> result.y;
	ss >> ch;
	ss >> result.width;
	ss >> ch;
	ss >> result.height;

	//获取图片列
	std::ifstream imagesFile;
	std::string fileName = imgPath + "images.txt";
	imagesFile.open(fileName);
	std::string text;
	std::vector<std::string> image_files;
	while (getline(imagesFile, text))
	{
		image_files.push_back(text);
	}
	cv::Mat image;
	//time
	int64 tic, toc;
	double timeRun = 0;


	//
	std::string savePath;
	std::string savePathFiles;
	if (1) {
		savePathFiles = staple.saveImage(imgSavePath);
	}

	for (unsigned int frame = 0; frame < image_files.size(); ++frame) {
		image = cv::imread(imgPath+image_files[frame]);
		tic = cv::getTickCount();
		if (frame == 0) {
			staple.tracker_staple_initialize(image, result);
			staple.tracker_staple_train(image, true);
		}
		else {
			result = staple.tracker_staple_update(image);
			staple.tracker_staple_train(image, false);
		}

		toc = cv::getTickCount() - tic;
		

		if (1) {
			cv::putText(image, std::to_string(frame + 1), cv::Point(20, 40), 6, 1,
				cv::Scalar(0, 255, 255), 2);
			cv::rectangle(image, result, cv::Scalar(0, 128, 255), 2);
			cv::imshow("STAPLE", image);

			char key = cv::waitKey(10);
			//保存结果
			if (1) {
				savePath = savePathFiles + "\\" + image_files[frame].substr(0, 8) + ".jpg";
				imwrite(savePath, image);
			}
			if (key == 27 || key == 'q' || key == 'Q')
				break;
		}
	}
	cv::destroyAllWindows();
	return 0;
}
