#pragma once
#include "tracker.h"
#ifndef _OPENCV_KCFTRACKER_HPP_
#define _OPENCV_KCFTRACKER_HPP_
#endif
using namespace std;

class newTracker :
	public tracker
{
private:
	bool _hogfeatures;
	bool _labfeatures;
	bool _colorhistfeatures;

	int size_patch[3];
	cv::Mat hann;
	cv::Size _tmpl_sz;
	float _scale;
	int _gaussian_size;

	//
	cv::Point_<float> _pos;
	cv::Size _target_sz;
	double _inner_padding;
	cv::Size _bg_area;
	cv::Size _fg_area;
	double _alpha;
	double _learning_rate_pwp;

	double _area_resize_factor;
	cv::Size _cf_response_size;

	cv::Size _norm_bg_area;
	cv::Size _norm_target_sz;
	cv::Size _norm_delta_area;
	cv::Size _norm_pwp_search_area;

	cv::Mat _im_patch_pwp;

	cv::Mat _bg_hist;
	cv::Mat _fg_hist;

protected:
	cv::Mat _alphaf;
	cv::Mat _prob;
	cv::Mat _tmpl;
	cv::Mat _num;
	cv::Mat _den;
	cv::Mat _labCentroids;

	int _bins;

	cv::Point2f detect(cv::Mat image, cv::Mat z, cv::Mat x, float &peak_value);
	void train(cv::Mat x, float train_interp_factor);
	cv::Mat gaussianCorrelation(cv::Mat x1, cv::Mat x2);
	cv::Mat createGaussianPeak(int sizey, int sizex);
	cv::Mat getFeatures(const cv::Mat & image, bool inithann, float scale_adjust = 1.0f);
	void createHanningMats();
	float subPixelPeak(float left, float center, float right);

	void initializeAllAreas(const cv::Mat &im, const cv::Rect &roi);
	void updateHistModel(bool new_model, cv::Mat &patch, double learning_rate_pwp);
	void getSubwindow(const cv::Mat &im, cv::Point_<float> centerCoor, cv::Size model_sz, cv::Size scaled_sz, cv::Mat &output);
	void mexResize(const cv::Mat &im, cv::Mat &output, cv::Size newsz, const char *method);
	void getColourMap(const cv::Mat &patch, cv::Mat& output);
	void getCenterLikelihood(const cv::Mat &object_likelihood, cv::Size m, cv::Mat& center_likelihood);
	void cropFilterResponse(const cv::Mat &response_cf, cv::Size response_size, cv::Mat& output);

public:
	float interp_factor; // linear interpolation factor for adaptation
	//float hog_interp_factor
	//float lab_interp_factor;
	//float cn_interp_factor;

	float sigma; // gaussian kernel bandwidth
	//float hog_sigma;
	//float lab_sigma;
	//float cn_sigma;

	float output_sigma_factor; // bandwidth of gaussian target
	//float hog_output_sigma_factor;
	//float lab_output_sigma_factor;
	//float cn_output_sigma_factor;

	float lambda; // regularization
	int cell_size; // HOG cell size
	int cell_sizeQ; // cell size^2, to avoid repeated operations
	float padding; // extra area surrounding the target
	int template_size; // template size
	float scale_step; // scale step for multi-scale estimation
	float scale_weight;  // to downweight detection scores of other scales for added stability

	newTracker(bool hog = true, bool lab = true, bool colorhistfeature = false, bool fixed_window = true, bool multiscale = true);
	virtual void init(const cv::Rect &roi, cv::Mat image);
	virtual cv::Rect update(cv::Mat image);

	std::string saveImage(string saveImgPath, bool hog, bool lab, bool colorhist, bool multiscale);
	//hog, lab, colorhist, multiscale
	void saveVideo(string saveImgPath, string saveVideoPath, string videoNameTemp, cv::Size imgSize, bool multiscale, bool hog, bool lab, bool colorhist);
	cv::Mat readYUVImage(string imagePath, cv::Size imgYUVSize);

	ofstream  f;
	double maxFHog = 0, maxColorHist = 0;

};

