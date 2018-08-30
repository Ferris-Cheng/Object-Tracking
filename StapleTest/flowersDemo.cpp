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

//static bool isstorage = true;

enum   InputFormatTypes { yuv, jpg };

std::vector<cv::Rect_<float>> getgroundtruth(std::string txt_file);

int main(int argc, char * argv[])
{
	bool isstorage = true;
	bool isSaveVideo = true;
	bool show_visualization = true;
	//
	STAPLE_TRACKER staple;
	//staple.isstorage = true;
	//staple.isSaveVideo = true;
	//staple.isusecolor = true;
	//staple.isusehog = true;
	//staple.scale_adaptation = true;
	//staple.show_visualization = true;

	//staple.isstorage = true;

	std::string PATH_IMG_TOPCV = "D:\\2-resources\\2.2-ObjcetTracking\\Test_case";
	std::string seq_name = "flowers4";
	
	int format = yuv;
	std::string imgPath = PATH_IMG_TOPCV + "\\" + seq_name + "\\";

	std::string imgSavePathTemp = "D:\\1-documents\\1.1-ObjectTracking\\1.1.4-CaseTest\\TestResults\\";
	std::string methodName = "staple";
	std::string imgSavePath = imgSavePathTemp + methodName;

	cv::Size imgYUVSize;
	imgYUVSize.width = 1440;
	imgYUVSize.height = 1080;

	//groundtruth
	std::ifstream initInfoFile;
	std::string fileName = imgPath + "groundtruth.txt";
	std::vector<cv::Rect_<float>> groundtruth_rect;
	groundtruth_rect = getgroundtruth(fileName);
	cv::Rect_<float> location = groundtruth_rect[0];

	//get picture  sequence
	std::ifstream imagesFile;
	fileName = imgPath + "images.txt";
	imagesFile.open(fileName);
	std::string text;
	std::vector<std::string> image_files;
	while (getline(imagesFile, text))
	{
		image_files.push_back(text);
	}

	cv::Mat image;
	std::vector<cv::Rect_<float>> result_rects;
	int64 tic, toc;
	double timeRun = 0;

	
	cv::Mat show_img;
	unsigned char* pYuvBuf = NULL;
	int bufLen = 0;

	std::string savePathFiles;
	if (1) {
		savePathFiles = staple.saveImage(imgSavePath);
	}

	for (unsigned int frame = 0; frame < image_files.size(); ++frame) {
		std::string savePath;
		std::string imgFinalPath = imgPath + image_files[frame];
		/*
		if (format == yuv) {
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
		}
		image = show_img;
		*/
		image = staple.readYUVImage(imgFinalPath, imgYUVSize);
		
		tic = cv::getTickCount();

		if (frame == 0) {
			staple.tracker_staple_initialize(image, location);
			staple.tracker_staple_train(image, true);
		}
		else {
			location = staple.tracker_staple_update(image);
			staple.tracker_staple_train(image, false);
		}

		toc = cv::getTickCount() - tic;
		timeRun += toc;
		result_rects.push_back(location);

		if (show_visualization) {

			cv::putText(image, std::to_string(frame + 1), cv::Point(20, 40), 6, 1,
				cv::Scalar(0, 255, 255), 2);
			//cv::rectangle(image, groundtruth_rect[frame], cv::Scalar(0, 255, 0), 2);
			cv::rectangle(image, location, cv::Scalar(0, 128, 255), 2);

			cv::imshow("STAPLE", image);
			//保存结果
			if (isstorage) {
				savePath = savePathFiles + "\\" + image_files[frame].substr(0, 8) + ".jpg";
				imwrite(savePath, image);
				/*
				std::time_t nowtime;
				nowtime = time(NULL);
				struct tm *local;
				local = localtime(&nowtime);
				ostringstream smonth, sday;
				std::string month, day;
				smonth << (local->tm_mon + 1);
				month = smonth.str();
				sday << local->tm_mday;
				day = sday.str();

				std::string storagePath = "D:\\1-documents\\1.1-ObjectTracking\\1.1.4-CaseTest\\TestResults\\";
				std::string filesName = month + "-" + day;
				
				storagePath += filesName+"-H-S";
				if (access(storagePath.c_str(), 0) == -1)
					int flag = mkdir(storagePath.c_str());
				storagePath = storagePath + "\\" + image_files[frame].substr(0, 8) + ".jpg";
				imwrite(storagePath, image);
				*/
			}
			char key = cv::waitKey(1);
			if (key == 27 || key == 'q' || key == 'Q')
				break;
		}
	}
	timeRun = timeRun / double(cv::getTickFrequency());
	double fps = double(result_rects.size()) / timeRun;
	std::cout << "fps:" << fps << std::endl;
	if (isSaveVideo) {
		if (isstorage) {
			//void saveVideo(string saveImgPath, string saveVideoPath, string videoNameTemp, cv::Size imgSize, bool hog, bool colorhist);
			staple.saveVideo(savePathFiles, imgSavePathTemp, methodName, imgYUVSize);
		}
		else {
			cout << "Error: not save image";
		}
	}
	cv::destroyAllWindows();

	return 0;

}

std::vector<cv::Rect_<float>> getgroundtruth(std::string txt_file)
{
	std::vector<cv::Rect_<float>> rects;
	std::ifstream initInfoFile;
	initInfoFile.open(txt_file);
	std::string Line;
	float initX, initY, initWidth, initHegiht;
	char ch;
	while (getline(initInfoFile, Line)) {
		std::istringstream ss(Line);
		ss >> initX, ss >> ch;
		ss >> initY, ss >> ch;
		ss >> initWidth, ss >> ch;
		ss >> initHegiht, ss >> ch;
		cv::Rect_<float> initRect = cv::Rect(initX, initY, initWidth, initHegiht);
		rects.push_back(initRect);
	}
	initInfoFile.close();
	return rects;
}
