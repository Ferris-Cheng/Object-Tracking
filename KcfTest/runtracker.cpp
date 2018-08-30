#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <string>
#include <numeric>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
//
#include<io.h>
#include<direct.h>
#include<time.h>
#include "kcftracker.hpp"
//#include <dirent.h>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]){

	if (argc > 5) return -1;
	bool isstorage = false;
	bool hog = true, fixedwindow = false, multiscale = true, lab = false, cn = false, gray = false;

	/*for(int i = 0; i < argc; i++){
		if ( strcmp (argv[i], "hog") == 0 )
			HOG = true;
		if ( strcmp (argv[i], "fixed_window") == 0 )
			FIXEDWINDOW = true;
		if ( strcmp (argv[i], "singlescale") == 0 )
			MULTISCALE = false;
		if ( strcmp (argv[i], "show") == 0 )
			SILENT = false;
		if ( strcmp (argv[i], "lab") == 0 ){
			LAB = true;
			HOG = true;
		}
		if ( strcmp (argv[i], "gray") == 0 )
			HOG = false;
	}*/

	//创建KCFTracker
	KCFTracker tracker(hog, fixedwindow, multiscale, lab, cn, gray);

	// Frame readed
	Mat frame;

	// Tracker results
	Rect result;

	int testVersion = 1;

	switch (testVersion) {
		//代码原始版本
	case 0: {
		// Read groundtruth for the 1st frame
		ifstream groundtruthFile;
		string groundtruth = "region.txt";
		groundtruthFile.open(groundtruth);
		string firstLine;
		getline(groundtruthFile, firstLine);
		groundtruthFile.close();

		istringstream ss(firstLine);

		// Read groundtruth like a dumb
		float x1, y1, x2, y2, x3, y3, x4, y4;
		char ch;
		ss >> x1;
		ss >> ch;
		ss >> y1;
		ss >> ch;
		ss >> x2;
		ss >> ch;
		ss >> y2;
		ss >> ch;
		ss >> x3;
		ss >> ch;
		ss >> y3;
		ss >> ch;
		ss >> x4;
		ss >> ch;
		ss >> y4;

		// Using min and max of X and Y for groundtruth rectangle
		float xMin = min(x1, min(x2, min(x3, x4)));
		float yMin = min(y1, min(y2, min(y3, y4)));
		float width = max(x1, max(x2, max(x3, x4))) - xMin;
		float height = max(y1, max(y2, max(y3, y4))) - yMin;


		// Read Images
		ifstream listFramesFile;
		string listFrames = "images.txt";
		listFramesFile.open(listFrames);
		string frameName;


		// Write Results
		ofstream resultsFile;
		string resultsPath = "output.txt";
		resultsFile.open(resultsPath);

		// Frame counter
		int nFrames = 0;


		while (getline(listFramesFile, frameName)) {
			frameName = frameName;

			// Read each frame from the list
			frame = imread(frameName, CV_LOAD_IMAGE_COLOR);

			// First frame, give the groundtruth to the tracker
			if (nFrames == 0) {
				tracker.init(Rect(xMin, yMin, width, height), frame);
				rectangle(frame, Point(xMin, yMin), Point(xMin + width, yMin + height), Scalar(0, 255, 255), 1, 8);
				resultsFile << xMin << "," << yMin << "," << width << "," << height << endl;
			}
			// Update
			else {
				result = tracker.update(frame);
				rectangle(frame, Point(result.x, result.y), Point(result.x + result.width, result.y + result.height), Scalar(0, 255, 255), 1, 8);
				resultsFile << result.x << "," << result.y << "," << result.width << "," << result.height << endl;
			}

			nFrames++;
		}
		resultsFile.close();
	}
		break;
		//测试flowers版本
	case 1: {
		std::string PATH_IMG_TOPCV = "D:\\2-resources\\2.2-ObjcetTracking\\Test_case";
		std::string seq_name = "flowers4";
		std::string imgPath = PATH_IMG_TOPCV + "\\" + seq_name + "\\";
		//获取第一帧的region
		ifstream groundtruthFile;
		string groundtruth = imgPath + "groundtruth.txt";
		groundtruthFile.open(groundtruth);
		string firstLine;
		getline(groundtruthFile, firstLine);
		groundtruthFile.close();

		istringstream ss(firstLine);
		// Read groundtruth like a dumb
		float x1, y1, width, height;
		char ch;
		ss >> x1;
		ss >> ch;
		ss >> y1;
		ss >> ch;
		ss >> width;
		ss >> ch;
		ss >> height;

		result.x = x1;
		result.y = y1;
		result.width = width;
		result.height = height;

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
		int64 tic, toc;
		double timeRun = 0;

		bool show_visualization = true;
		cv::Mat show_img;
		unsigned char* pYuvBuf = NULL;
		int bufLen = 0;

		for (unsigned int frame = 0; frame < image_files.size(); ++frame) {
			tic = cv::getTickCount();
			//读取yuv图像
			std::string imgFinalPath = imgPath + image_files[frame];
			FILE* pFileIn = fopen(imgFinalPath.c_str(), "r");
			//int framewidth = 768;
			//int frameheight = 576;
			int framewidth = 1440;
			int frameheight = 1080;
			bufLen = framewidth * frameheight * 3 / 2;
			pYuvBuf = new unsigned char[bufLen]();
			fread(pYuvBuf, bufLen * sizeof(unsigned char), 1, pFileIn);
			cv::Mat imageTemp(frameheight * 3 / 2, framewidth, CV_8UC1, pYuvBuf);
			cvtColor(imageTemp, show_img, CV_YUV2BGR_NV21);
			fclose(pFileIn);
			//imageS = pYuvBuf;
			image = show_img;

			if (frame == 0) {
				tracker.init(result, image);
			}
			else {
				result = tracker.update(image);
			}
			toc = cv::getTickCount() - tic;
			timeRun += toc;
			if (show_visualization) {
				cv::putText(image, std::to_string(frame + 1), cv::Point(20, 40), 6, 1,
					cv::Scalar(0, 255, 255), 2);
				//cv::rectangle(image, groundtruth_rect[frame], cv::Scalar(0, 255, 0), 2);
				cv::rectangle(image, Point(result.x, result.y), Point(result.x + result.width, result.y + result.height), cv::Scalar(0, 128, 255), 2);
				cv::imshow("KCF", image);
				//保存结果
				if (isstorage) {
					std::time_t nowtime;
					nowtime = time(NULL);
					struct tm *local;
					local = localtime(&nowtime);
					ostringstream smonth, sday;
					std::string month, day;
					smonth << (local->tm_mon+1);
					month = smonth.str();
					sday << local->tm_mday;
					day = sday.str();

					std::string storagePath = "D:\\1-documents\\1.1-ObjectTracking\\1.1.4-CaseTest\\TestResults\\";
					std::string filesName = month + "-" + day;
					if (hog)
						filesName += "-H";
					if (multiscale)
						filesName += "-M";
					if(lab)
						filesName += "-L";
					if(cn)
						filesName += "-C";
					if(gray)
						filesName += "-G";
					storagePath += filesName;
					if (access(storagePath.c_str(), 0) == -1)
						int flag = mkdir(storagePath.c_str());
					storagePath = storagePath+"\\"+ image_files[frame].substr(0, 8) + ".jpg";
					imwrite(storagePath, image);
				}
				char key = cv::waitKey(1);
				if (key == 27 || key == 'q' || key == 'Q')
					break;
			}
		}
		timeRun = timeRun / double(cv::getTickFrequency());
		double fps = double(175) / timeRun;
		std::cout << "fps:" << fps << std::endl;
		cv::destroyAllWindows();

		return 0;
	}
	}

}
