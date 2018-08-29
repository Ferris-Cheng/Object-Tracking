//测试程序，用于测试YUV、JPG等图片集
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

std::string rName(bool multiscale,bool hog, bool lab, bool colorhist) {
	std::string rName = "";
	/*
	if (multiscale) {
		rName += "M";
	}
	*/
	if (hog) {
		rName += "-Hog";
	}
	if (lab) {
		rName += "-Lab";
	}
	if (colorhist) {
		rName += "-CHist";
	}
	return rName;

}

int main(int argc, char* argv[]) {

	bool hog = true;
	bool lab = false;
	bool colorhist = true;

	bool fixedwindow = true;
	bool multiscale = true;

	bool isSaveImage = true;
	bool isSaveVideo = false;
	bool show_visualization = true;

	newTracker tracker(hog, lab, colorhist, fixedwindow, multiscale);   //create tracker
	

	//数据集路径
	std::string PATH_IMG_TOPCV = "D:\\2-resources\\2.2-ObjcetTracking\\OTB";
	std::string seq_name = "Skater";
	std::string imgPath = PATH_IMG_TOPCV + "\\" + seq_name + "\\";

	//图片和视频保存路径
	std::string imgSavePathTemp = "D:\\1-documents\\1.1-ObjectTracking\\1.1.4-CaseTest\\TestResults\\";
	std::string methodName = "newTrackerOTB";
	methodName += seq_name;
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

	std::string savePathFiles;
	if (isSaveImage) {
		savePathFiles = tracker.saveImage(imgSavePath, hog, lab, colorhist, multiscale);
	}

	int imageType = 1; //0：YUV    1：JPG

	switch (imageType) {
	case 0: {
		cv::Size imgYUVSize;

		//just for D:\2-resources\2.2-ObjcetTracking\Test_case\flowers4
		imgYUVSize.width = 1440;
		imgYUVSize.height = 1080;
		for (unsigned int frame = 0; frame < image_files.size(); ++frame) {
			std::string savePath;
			tic = cv::getTickCount();
			std::string imgFinalPath = imgPath + image_files[frame];
			//读取yuv图像
			image = tracker.readYUVImage(imgFinalPath, imgYUVSize);

			if (frame == 0)
				tracker.init(result, image);
			else
				result = tracker.update(image);

			toc = cv::getTickCount() - tic;
			timeRun += toc;
			if (show_visualization) {
				cv::putText(image, std::to_string(frame + 1), cv::Point(20, 40), 6, 1,cv::Scalar(0, 255, 255), 2);
				cv::rectangle(image, Point(result.x, result.y), Point(result.x + result.width, result.y + result.height), cv::Scalar(0, 128, 255), 2);
				cv::imshow("newtracker", image);

				//保存结果
				if (isSaveImage) {
					savePath = savePathFiles + "\\" + image_files[frame].substr(0, 8) + ".jpg";
					imwrite(savePath, image);
				}

				char key = cv::waitKey(1);
				if (key == 27 || key == 'q' || key == 'Q')
					break;
			}
		}

		timeRun = timeRun / double(cv::getTickFrequency());
		double fps = double(399) / timeRun;
		std::cout << "fps:" << fps << std::endl;
		if (isSaveVideo) {
			if (isSaveImage) {
				//saveVideo(string saveImgPath, string saveVideoPath, string videoNameTemp, cv::Size imgSize)
				tracker.saveVideo(savePathFiles, imgSavePathTemp, methodName, imgYUVSize, multiscale, hog, lab, colorhist);
			}
			else {
				cout << "Error: not save image";
			}
		}
		cv::destroyAllWindows();
		return 0;
	}
		break;
	case 1: {
		std::string savePath;
		std::string imgFinalPath;
		for (unsigned int frame = 0; frame < image_files.size(); ++frame) {
			
			tic = cv::getTickCount();
			imgFinalPath = imgPath + image_files[frame];
			image = imread(imgFinalPath);

			if (frame == 0)
				tracker.init(result, image);
			else
				result = tracker.update(image);

			toc = cv::getTickCount() - tic;
			timeRun += toc;

			if (show_visualization) {
				cv::putText(image, rName(multiscale, hog, lab, colorhist) + std::to_string(frame + 1), cv::Point(20, 40), 2, 1, cv::Scalar(0, 255, 255), 2);
				cv::rectangle(image, Point(result.x, result.y), Point(result.x + result.width, result.y + result.height), cv::Scalar(0, 128, 255), 2);
				cv::imshow(seq_name, image);

				//保存结果
				if (isSaveImage) {
					savePath = savePathFiles + "\\" + image_files[frame].substr(0, 8) + ".jpg";
					imwrite(savePath, image);
				}

				char key = cv::waitKey(1);
				if (key == 27 || key == 'q' || key == 'Q')
					break;
			}
		}

		timeRun = timeRun / double(cv::getTickFrequency());
		double fps = double(image_files.size()) / timeRun;
		std::cout << "fps:" << fps << std::endl;
		if (isSaveVideo) {
			cv::Size tSize;
			tSize.width = image.cols;
			tSize.height = image.rows;
			if (isSaveImage) {
				//saveVideo(string saveImgPath, string saveVideoPath, string videoNameTemp, cv::Size imgSize)
				tracker.saveVideo(savePathFiles, imgSavePathTemp, methodName, tSize, multiscale, hog, lab, colorhist);
			}
			else {
				cout << "Error: not save image";
			}
		}
		cv::destroyAllWindows();
		return 0;
	}
	default:
		cout << "error" << endl;
		break;
	}
}
