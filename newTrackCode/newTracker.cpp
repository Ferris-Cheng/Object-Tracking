#ifndef _KCFTRACKER_HEADERS
#include "newTracker.h"
#include "..\\ffttools.hpp"
#include "..\\fhog.hpp"
#include "..\\labdata.hpp"
#include "..\\recttools.hpp"
#include<io.h>
#include<direct.h>
#include<opencv2\opencv.hpp>
#endif
using namespace std;
using namespace cv;

static int testNumber = 1;
//static int modelUpdate = 0;

newTracker::newTracker(bool hog, bool lab, bool colorhist, bool fixed_window, bool multiscale) {
	lambda = 0.001;
	padding = 2.5;
	output_sigma_factor = 0.125;
	
	f.open("D:\\1-documents\\1.1-ObjectTracking\\1.1.4-CaseTest\\tRes.txt", ios::trunc);
	//
	
	if (hog) {
		_hogfeatures = true;
		interp_factor = 0.012;

		sigma = 0.6;

		cell_size = 4;
		cell_sizeQ = cell_size * cell_size;

		if (lab) {
			_labfeatures = true;

			interp_factor = 0.005;
			sigma = 0.4;

			output_sigma_factor = 0.1;

			_labfeatures = true;
			cell_sizeQ = cell_size * cell_size;

			_labCentroids = cv::Mat(nClusters, 3, CV_32FC1, &dataClusters);
		}
	}
	else {  //RAW
		_hogfeatures = false;

		interp_factor = 0.075;
		sigma = 0.2;
		cell_size = 1;

		if (lab) {
			printf("Lab features are only used with HOG features.\n");
			_labfeatures = false;
		}
	}

	if (colorhist) {
		_colorhistfeatures = true;
		//_learning_rate_pwp = 0.04;
		_learning_rate_pwp = 0.004;//czx
		_alpha = 0.4;
		_bins = 32;
		_inner_padding = 0.98;//取90%为fg
	}

	if (multiscale) { // multiscale
		//template_size = 96;
		template_size = 150;
		scale_step = 1.05;
		scale_weight = 0.95;
		if (!fixed_window) {
			//printf("Multiscale does not support non-fixed window.\n");
			fixed_window = true;
		}
	}
	else if (fixed_window) {  // fit correction without multiscale
		//template_size = 96;
		template_size = 150;
		scale_step = 1;
	}
	else {
		template_size = 1;
		scale_step = 1;
	}
	
}

void newTracker::initializeAllAreas(const cv::Mat &im, const cv::Rect &roi) {

	_target_sz.width = round(roi.width);
	_target_sz.height = round(roi.height);

	_pos.x = roi.x + roi.width / 2.0;
	_pos.y = roi.y + roi.height / 2.0;

	//double avg_dim = (_target_sz.width + _target_sz.height) / 2.0;
	_bg_area.width = round(_target_sz.width*padding);
	_bg_area.height = round(_target_sz.height*padding);

	_fg_area.width = round(_target_sz.width * _inner_padding);
	_fg_area.height = round(_target_sz.height * _inner_padding);

	cv::Size imsize = im.size();
	_bg_area.width = std::min(_bg_area.width, imsize.width - 1);
	_bg_area.height = std::min(_bg_area.height, imsize.height - 1);

	_bg_area.width = _bg_area.width - (_bg_area.width - _target_sz.width) % 2;
	_bg_area.height = _bg_area.height - (_bg_area.height - _target_sz.height) % 2;

	_fg_area.width = _fg_area.width + (_bg_area.width - _fg_area.width) % 2;
	_fg_area.height = _fg_area.height + (_bg_area.height - _fg_area.width) % 2;

	if (_bg_area.width > _bg_area.height) {
		_area_resize_factor = _bg_area.width / (float)template_size;
	}
	else {
		_area_resize_factor = _bg_area.height / (float)template_size;
	}
	_norm_bg_area.width = round(_bg_area.width / _area_resize_factor);
	_norm_bg_area.height = round(_bg_area.height / _area_resize_factor);
	//cellSize的倍数
	_norm_bg_area.width = ((int)(_norm_bg_area.width / (2 * cell_size))) * 2 * cell_size;
	_norm_bg_area.height = ((int)(_norm_bg_area.height / (2 * cell_size))) * 2 * cell_size;

	_norm_target_sz.width = round(roi.width / _area_resize_factor);
	_norm_target_sz.height = round(roi.height / _area_resize_factor);

	/**/
	cv::Size norm_pad;

	norm_pad.width = floor((_norm_bg_area.width - _norm_target_sz.width) / 2.0);
	norm_pad.height = floor((_norm_bg_area.height - _norm_target_sz.height) / 2.0);

	int radius = floor(fmin(norm_pad.width, norm_pad.height));

	_norm_delta_area = cv::Size((2 * radius + 1), (2 * radius + 1));

	_norm_pwp_search_area.width = _norm_target_sz.width + _norm_delta_area.width - 1;
	_norm_pwp_search_area.height = _norm_target_sz.height + _norm_delta_area.height - 1;
}
//计算特征
cv::Mat newTracker::getFeatures(const cv::Mat & image, bool inithann, float scale_adjust)
{
	cv::Rect extracted_roi;
	//cv::Size temp_size;

	float cx = _roi.x + _roi.width / 2;
	float cy = _roi.y + _roi.height / 2;

	if (inithann) {
		if (_hogfeatures) {
			_tmpl_sz.width = _norm_bg_area.width + cell_size * 2;
			_tmpl_sz.height = _norm_bg_area.height + cell_size * 2;
		}
		else {
			_tmpl_sz.width = (_norm_bg_area.width / 2) * 2;
			_tmpl_sz.height = (_norm_bg_area.height / 2) * 2;
		}
	}

	extracted_roi.width = scale_adjust * _area_resize_factor*_tmpl_sz.width;
	extracted_roi.height = scale_adjust * _area_resize_factor*_tmpl_sz.height;

	//extracted_roi.width = scale_adjust * _scale * _tmpl_sz.width;
	//extracted_roi.height = scale_adjust * _scale * _tmpl_sz.height;

	// center roi with new size
	extracted_roi.x = cx - extracted_roi.width / 2;
	extracted_roi.y = cy - extracted_roi.height / 2;

	cv::Mat FeaturesMap;
	cv::Mat z = RectTools::subwindow(image, extracted_roi, cv::BORDER_REPLICATE);

	if (z.cols != _tmpl_sz.width || z.rows != _tmpl_sz.height) {
		cv::resize(z, z, _tmpl_sz);
	}


	size_patch[2] = 0;

	if (_hogfeatures) {
		IplImage z_ipl = z;
		CvLSVMFeatureMapCaskade *map;
		getFeatureMaps(&z_ipl, cell_size, &map);
		normalizeAndTruncate(map, 0.2f);
		PCAFeatureMaps(map);
		size_patch[0] = map->sizeY;
		size_patch[1] = map->sizeX;
		size_patch[2] = map->numFeatures;

		FeaturesMap = cv::Mat(cv::Size(map->numFeatures, map->sizeX*map->sizeY), CV_32F, map->map);  // Procedure do deal with cv::Mat multichannel bug
		FeaturesMap = FeaturesMap.t();
		freeFeatureMapObject(&map);

		// Lab features
		if (_labfeatures) {
			cv::Mat imgLab;
			cvtColor(z, imgLab, CV_BGR2Lab);
			unsigned char *input = (unsigned char*)(imgLab.data);

			// Sparse output vector
			cv::Mat outputLab = cv::Mat(_labCentroids.rows, size_patch[0] * size_patch[1], CV_32F, float(0));

			int cntCell = 0;
			// Iterate through each cell
			for (int cY = cell_size; cY < z.rows - cell_size; cY += cell_size) {
				for (int cX = cell_size; cX < z.cols - cell_size; cX += cell_size) {
					// Iterate through each pixel of cell (cX,cY)
					for (int y = cY; y < cY + cell_size; ++y) {
						for (int x = cX; x < cX + cell_size; ++x) {
							// Lab components for each pixel
							float l = (float)input[(z.cols * y + x) * 3];
							float a = (float)input[(z.cols * y + x) * 3 + 1];
							float b = (float)input[(z.cols * y + x) * 3 + 2];

							// Iterate trough each centroid
							float minDist = FLT_MAX;
							int minIdx = 0;
							float *inputCentroid = (float*)(_labCentroids.data);
							for (int k = 0; k < _labCentroids.rows; ++k) {
								float dist = ((l - inputCentroid[3 * k]) * (l - inputCentroid[3 * k]))
									+ ((a - inputCentroid[3 * k + 1]) * (a - inputCentroid[3 * k + 1]))
									+ ((b - inputCentroid[3 * k + 2]) * (b - inputCentroid[3 * k + 2]));
								if (dist < minDist) {
									minDist = dist;
									minIdx = k;
								}
							}
							// Store result at output
							outputLab.at<float>(minIdx, cntCell) += 1.0 / cell_sizeQ;
							//((float*) outputLab.data)[minIdx * (size_patch[0]*size_patch[1]) + cntCell] += 1.0 / cell_sizeQ; 
						}
					}
					cntCell++;
				}
			}
			// Update size_patch[2] and add features to FeaturesMap
			size_patch[2] += _labCentroids.rows;
			FeaturesMap.push_back(outputLab);
		}
	}
	else {
		FeaturesMap = RectTools::getGrayImage(z);
		FeaturesMap -= (float) 0.5; // In Paper;
		size_patch[0] = z.rows;
		size_patch[1] = z.cols;
		size_patch[2] = 1;
	}
	/*
	if (_hogfeatures) {
		cv::Mat hogFeatureMap;
		IplImage z_ipl = z;
		CvLSVMFeatureMapCaskade *map;
		getFeatureMaps(&z_ipl, cell_size, &map);
		normalizeAndTruncate(map, 0.2f);
		PCAFeatureMaps(map);
		size_patch[0] = map->sizeY;
		size_patch[1] = map->sizeX;
		size_patch[2] += map->numFeatures;

		hogFeatureMap = cv::Mat(cv::Size(map->numFeatures, map->sizeX*map->sizeY), CV_32F, map->map);  // Procedure do deal with cv::Mat multichannel bug
		hogFeatureMap = hogFeatureMap.t();

		freeFeatureMapObject(&map);

		FeaturesMap.push_back(hogFeatureMap);
	}
	if (_labfeatures&&(z.channels() == 3)) {
		cv::Mat imgLab;
		cvtColor(z, imgLab, CV_BGR2Lab);
		unsigned char *input = (unsigned char*)(imgLab.data);
		// Sparse output vector
		cv::Mat outputLab = cv::Mat(_labCentroids.rows, size_patch[0] * size_patch[1], CV_32F, float(0));
		int cntCell = 0;
		// Iterate through each cell
		for (int cY = cell_size; cY < z.rows - cell_size; cY += cell_size) {
			for (int cX = cell_size; cX < z.cols - cell_size; cX += cell_size) {
				// Iterate through each pixel of cell (cX,cY)
				for (int y = cY; y < cY + cell_size; ++y) {
					for (int x = cX; x < cX + cell_size; ++x) {
						// Lab components for each pixel
						float l = (float)input[(z.cols * y + x) * 3];
						float a = (float)input[(z.cols * y + x) * 3 + 1];
						float b = (float)input[(z.cols * y + x) * 3 + 2];

						// Iterate trough each centroid
						float minDist = FLT_MAX;
						int minIdx = 0;
						float *inputCentroid = (float*)(_labCentroids.data);
						for (int k = 0; k < _labCentroids.rows; ++k) {
							float dist = ((l - inputCentroid[3 * k]) * (l - inputCentroid[3 * k]))
								+ ((a - inputCentroid[3 * k + 1]) * (a - inputCentroid[3 * k + 1]))
								+ ((b - inputCentroid[3 * k + 2]) * (b - inputCentroid[3 * k + 2]));
							if (dist < minDist) {
								minDist = dist;
								minIdx = k;
							}
						}
						// Store result at output
						outputLab.at<float>(minIdx, cntCell) += 1.0 / cell_sizeQ;
					}
				}
				cntCell++;
			}
		}
		// Update size_patch[2] and add features to FeaturesMap
		size_patch[2] += _labCentroids.rows;
		FeaturesMap.push_back(outputLab);
	}
	if (_cnfeatures) {
		cv::cvtColor(z, z, CV_BGR2GRAY);
		z.convertTo(z, CV_32F, 1 / 255.f);
		//z -= (float)0.5;
		//unsigned char *input = (unsigned char*)(z.data);
		cv::Mat cnFeatureMap = cv::Mat(1, size_patch[0] * size_patch[1], CV_32F, float(0));
		int cntCell = 0;
		float tempSum = 0;
		//float temp = z.at<float>(1, 1);
		// Iterate through each cell
		for (int cY = cell_size; cY < z.rows - cell_size; cY += cell_size) {
			for (int cX = cell_size; cX < z.cols - cell_size; cX += cell_size) {
				tempSum = 0;
				// Iterate through each pixel of cell (cX,cY)
				for (int y = cY; y < cY + cell_size; ++y) {
					for (int x = cX; x < cX + cell_size; ++x) {
						tempSum += z.at<float>(y, x);
					}
				}
				cnFeatureMap.at<float>(0, cntCell++) = tempSum / 16;
			}
		}
		// Update size_patch[2] and add features to FeaturesMap
		size_patch[2] += 1;
		FeaturesMap.push_back(cnFeatureMap);
	}
	*/
	if (inithann) {
		createHanningMats();
	}
	FeaturesMap = hann.mul(FeaturesMap);
	return FeaturesMap;
}
// Initialize tracker 
void newTracker::init(const cv::Rect &roi, cv::Mat image)
{
	_roi = roi;
	assert(roi.width >= 0 && roi.height >= 0);

	initializeAllAreas(image,roi);

	if (_colorhistfeatures) {
		cv::Mat patch_padded;
		getSubwindow(image, _pos, _norm_bg_area, _bg_area, patch_padded);
		updateHistModel(true, patch_padded,0.04);
	}

	_tmpl = getFeatures(image, 1);
	_prob = createGaussianPeak(size_patch[0], size_patch[1]);

	_alphaf = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
	train(_tmpl, 1.0); // train with initial frame
}

void newTracker::getSubwindow(const cv::Mat &im, cv::Point_<float> centerCoor, cv::Size model_sz, cv::Size scaled_sz, cv::Mat &output) {
	cv::Size sz = scaled_sz; // scale adaptation

							 // make sure the size is not to small
	sz.width = fmax(sz.width, 2);
	sz.height = fmax(sz.height, 2);

	cv::Mat subWindow;

	// xs = round(pos(2) + (1:sz(2)) - sz(2)/2);
	// ys = round(pos(1) + (1:sz(1)) - sz(1)/2);

	cv::Point lefttop(
		std::min(im.cols - 1, std::max(-sz.width + 1, int(centerCoor.x + 1 - sz.width / 2.0 + 0.5))),
		std::min(im.rows - 1, std::max(-sz.height + 1, int(centerCoor.y + 1 - sz.height / 2.0 + 0.5)))
	);

	cv::Point rightbottom(
		std::max(0, int(lefttop.x + sz.width - 1)),
		std::max(0, int(lefttop.y + sz.height - 1))
	);

	cv::Point lefttopLimit(
		std::max(lefttop.x, 0),
		std::max(lefttop.y, 0)
	);
	cv::Point rightbottomLimit(
		std::min(rightbottom.x, im.cols - 1),
		std::min(rightbottom.y, im.rows - 1)
	);

	rightbottomLimit.x += 1;
	rightbottomLimit.y += 1;
	cv::Rect roiRect(lefttopLimit, rightbottomLimit);

	im(roiRect).copyTo(subWindow);

	int top = lefttopLimit.y - lefttop.y;
	int bottom = rightbottom.y - rightbottomLimit.y + 1;
	int left = lefttopLimit.x - lefttop.x;
	int right = rightbottom.x - rightbottomLimit.x + 1;

	cv::copyMakeBorder(subWindow, subWindow, top, bottom, left, right, cv::BORDER_REPLICATE);

	// imresize(subWindow, output, model_sz, 'bilinear', 'AntiAliasing', false)
	mexResize(subWindow, output, model_sz, "auto");
}

void newTracker::mexResize(const cv::Mat &im, cv::Mat &output, cv::Size newsz, const char *method) {
	int interpolation = cv::INTER_LINEAR;

	cv::Size sz = im.size();

	if (!strcmp(method, "antialias")) {
		interpolation = cv::INTER_AREA;
	}
	else if (!strcmp(method, "linear")) {
		interpolation = cv::INTER_LINEAR;
	}
	else if (!strcmp(method, "auto")) {
		if (newsz.width > sz.width) { // xxx
			interpolation = cv::INTER_LINEAR;
		}
		else {
			interpolation = cv::INTER_AREA;
		}
	}
	else {
		assert(0);
		return;
	}

	resize(im, output, newsz, 0, 0, interpolation);
}

void newTracker::updateHistModel(bool new_model, cv::Mat &patch, double learning_rate_pwp) {
	cv::Size pad_offset1;
	// we constrained the difference to be mod2, so we do not have to round here
	pad_offset1.width = (_bg_area.width - _target_sz.width) / 2;
	pad_offset1.height = (_bg_area.height - _target_sz.height) / 2;

	// difference between bg_area and target_sz has to be even
	if (
		(
		(pad_offset1.width == round(pad_offset1.width)) &&
			(pad_offset1.height != round(pad_offset1.height))
			) ||
			(
		(pad_offset1.width != round(pad_offset1.width)) &&
				(pad_offset1.height == round(pad_offset1.height))
				)) {
		assert(0);
	}

	pad_offset1.width = fmax(pad_offset1.width, 1);
	pad_offset1.height = fmax(pad_offset1.height, 1);

	//std::cout << "pad_offset1 " << pad_offset1 << std::endl;

	cv::Mat bg_mask(_bg_area, CV_8UC1, cv::Scalar(1)); // init bg_mask

													  // xxx: bg_mask(pad_offset1(1)+1:end-pad_offset1(1), pad_offset1(2)+1:end-pad_offset1(2)) = false;

	cv::Rect pad1_rect(
		pad_offset1.width,
		pad_offset1.height,
		_bg_area.width - 2 * pad_offset1.width,
		_bg_area.height - 2 * pad_offset1.height
	);

	bg_mask(pad1_rect) = false;

	////////////////////////////////////////////////////////////////////////
	cv::Size pad_offset2;

	// we constrained the difference to be mod2, so we do not have to round here
	pad_offset2.width = (_bg_area.width - _fg_area.width) / 2;
	pad_offset2.height = (_bg_area.height - _fg_area.height) / 2;

	// difference between bg_area and fg_area has to be even
	if (
		(
		(pad_offset2.width == round(pad_offset2.width)) &&
			(pad_offset2.height != round(pad_offset2.height))
			) ||
			(
		(pad_offset2.width != round(pad_offset2.width)) &&
				(pad_offset2.height == round(pad_offset2.height))
				)) {
		assert(0);
	}

	pad_offset2.width = fmax(pad_offset2.width, 1);
	pad_offset2.height = fmax(pad_offset2.height, 1);

	//std::cout << "pad_offset2 " << pad_offset2 << std::endl;

	cv::Mat fg_mask(_bg_area, CV_8UC1, cv::Scalar(0)); // init fg_mask

													  // xxx: fg_mask(pad_offset2(1)+1:end-pad_offset2(1), pad_offset2(2)+1:end-pad_offset2(2)) = true;

	cv::Rect pad2_rect(
		pad_offset2.width,
		pad_offset2.height,
		_bg_area.width - 2 * pad_offset2.width,
		_bg_area.height - 2 * pad_offset2.height
	);

	fg_mask(pad2_rect) = true;
	////////////////////////////////////////////////////////////////////////

	cv::Mat fg_mask_new;
	cv::Mat bg_mask_new;

	mexResize(fg_mask, fg_mask_new, _norm_bg_area, "auto");
	mexResize(bg_mask, bg_mask_new, _norm_bg_area, "auto");

	int imgCount = 1;
	int dims = 3;
	const int sizes[] = { _bins, _bins, _bins };
	const int channels[] = { 0, 1, 2 };
	float bRange[] = { 0, 256 };
	float gRange[] = { 0, 256 };
	float rRange[] = { 0, 256 };
	const float *ranges[] = { bRange, gRange, rRange };

	// (TRAIN) BUILD THE MODEL
	if (new_model) {
		cv::calcHist(&patch, imgCount, channels, bg_mask_new, _bg_hist, dims, sizes, ranges);
		cv::calcHist(&patch, imgCount, channels, fg_mask_new, _fg_hist, dims, sizes, ranges);

		int bgtotal = cv::countNonZero(bg_mask_new);
		(bgtotal == 0) && (bgtotal = 1);
		_bg_hist = _bg_hist / bgtotal;

		int fgtotal = cv::countNonZero(fg_mask_new);
		(fgtotal == 0) && (fgtotal = 1);
		_fg_hist = _fg_hist / fgtotal;
	}
	else { // update the model
		cv::MatND bg_hist_tmp;
		cv::MatND fg_hist_tmp;

		cv::calcHist(&patch, imgCount, channels, bg_mask_new, bg_hist_tmp, dims, sizes, ranges);
		cv::calcHist(&patch, imgCount, channels, fg_mask_new, fg_hist_tmp, dims, sizes, ranges);

		int bgtotal = cv::countNonZero(bg_mask_new);
		(bgtotal == 0) && (bgtotal = 1);
		bg_hist_tmp = bg_hist_tmp / bgtotal;

		int fgtotal = cv::countNonZero(fg_mask_new);
		(fgtotal == 0) && (fgtotal = 1);
		fg_hist_tmp = fg_hist_tmp / fgtotal;

		// xxx
		_bg_hist = (1 - learning_rate_pwp)*_bg_hist + learning_rate_pwp * bg_hist_tmp;
		_fg_hist = (1 - learning_rate_pwp)*_fg_hist + learning_rate_pwp * fg_hist_tmp;
	}
}
// train tracker with a single image
void newTracker::train(cv::Mat x, float train_interp_factor)
{
	using namespace FFTTools;

	cv::Mat k = gaussianCorrelation(x, x);
	cv::Mat alphaf = complexDivision(_prob, (fftd(k) + lambda));

	_tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor)* x;
	_alphaf = (1 - train_interp_factor) * _alphaf + (train_interp_factor)* alphaf;
}

// Update position based on the new frame
cv::Rect newTracker::update(cv::Mat image)
{
	if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 1;
	if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 1;
	if (_roi.x >= image.cols - 1) _roi.x = image.cols - 2;
	if (_roi.y >= image.rows - 1) _roi.y = image.rows - 2;

	float cx = _roi.x + _roi.width / 2.0f;
	float cy = _roi.y + _roi.height / 2.0f;


	float peak_value;
	cv::Point2f res = detect(image, _tmpl, getFeatures(image, 0, 1.0f), peak_value);

	if (scale_step != 1) {
		// Test at a smaller _scale
		float new_peak_value;
		cv::Point2f new_res = detect(image, _tmpl, getFeatures(image, 0, 1.0f / scale_step), new_peak_value);
		//_area_resize_factor=_scale
		if (scale_weight * new_peak_value > peak_value) {
			res = new_res;
			peak_value = new_peak_value;
			_area_resize_factor /= scale_step;
			_roi.width /= scale_step;
			_roi.height /= scale_step;
		}

		// Test at a bigger _scale
		new_res = detect(image, _tmpl, getFeatures(image, 0, scale_step), new_peak_value);

		if (scale_weight * new_peak_value > peak_value) {
			res = new_res;
			peak_value = new_peak_value;
			_area_resize_factor *= scale_step;
			_roi.width *= scale_step;
			_roi.height *= scale_step;
		}
	}

	// Adjust by cell size and _scale
	_roi.x = cx - _roi.width / 2.0f + ((float)res.x * _area_resize_factor);
	_roi.y = cy - _roi.height / 2.0f + ((float)res.y * _area_resize_factor);

	if (_roi.x >= image.cols - 1) _roi.x = image.cols - 1;
	if (_roi.y >= image.rows - 1) _roi.y = image.rows - 1;
	if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 2;
	if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 2;

	assert(_roi.width >= 0 && _roi.height >= 0);
	cv::Mat x = getFeatures(image, 0);

	_pos.x = _roi.x + _roi.width / 2.0;
	_pos.y = _roi.y + _roi.height / 2.0;

	if (_colorhistfeatures) {
		cv::Mat patch_padded;
		getSubwindow(image, _pos, _norm_bg_area, _bg_area, patch_padded);
		updateHistModel(false, patch_padded, _learning_rate_pwp);
	}


	//modelUpdate++;

	train(x, interp_factor);

	return _roi;
}

void newTracker::getColourMap(const cv::Mat &patch, cv::Mat& output) {
	cv::Size sz = patch.size();
	int h = sz.height;
	int w = sz.width;

	int bin_width = 256 / _bins;
	float probg;
	float profg;
	float *P_O = new float[w*h];

	for (int i = 0; i < w; i++)
		for (int j = 0; j < h; j++) {
			cv::Vec3b p = patch.at<cv::Vec3b>(j, i);

			int b1 = floor(p[0] / bin_width);
			int b2 = floor(p[1] / bin_width);
			int b3 = floor(p[2] / bin_width);

			float* histd;

			histd = (float*)_bg_hist.data;
			probg = histd[b1*_bins *_bins + b2 * _bins + b3];

			histd = (float*)_fg_hist.data;
			profg = histd[b1*_bins*_bins + b2 * _bins + b3];

			// xxx
			P_O[j*w + i] = profg / (profg + probg);
			isnan(P_O[j*w + i]) && (P_O[j*w + i] = 0.0);
			
		}

	output = cv::Mat(h, w, CV_32FC1, P_O).clone();

	//myEnd
	delete[] P_O;
}

void newTracker::getCenterLikelihood(const cv::Mat &object_likelihood, cv::Size m, cv::Mat& center_likelihood) {
	cv::Size sz = object_likelihood.size();
	int h = sz.height;
	int w = sz.width;
	int n1 = w - m.width + 1;
	int n2 = h - m.height + 1;
	int area = m.width * m.height;

	cv::Mat temp;

	// integral images
	cv::integral(object_likelihood, temp);

	float *CENTER_LIKELIHOOD = new float[n1*n2];
	//*原代码中计算CENTER_LIKELIHOOD，有错误，多计算了一行一列，并且偏移了；比如CENTER_LIKELIHOOD[0]应该计算的是temp的1:75 1:75，实际计算的是0:75 0:75
	for (int i = 0; i < n1; i++)
		for (int j = 0; j < n2; j++) {
			CENTER_LIKELIHOOD[j*n1 + i]
				= (temp.at<double>(j, i) + temp.at<double>(j + m.height, i + m.width) - temp.at<double>(j, i + m.width) - temp.at<double>(j + m.height, i)) / area;
		}
	/*
	for (int i = 0; i < n1; i++)
		for (int j = 0; j < n2; j++) {
			CENTER_LIKELIHOOD[j*n1 + i]
				= (temp.at<double>(j+1, i+1) + temp.at<double>(j + m.height, i + m.width) - temp.at<double>(j+1, i + m.width) - temp.at<double>(j + m.height, i+1)) / area;
		}
	*/
	// SAT = integralImage(object_likelihood);
	// i = 1:n1;
	// j = 1:n2;
	// center_likelihood = (SAT(i,j) + SAT(i+m(1), j+m(2)) - SAT(i+m(1), j) - SAT(i, j+m(2))) / prod(m);

	center_likelihood = cv::Mat(n2, n1, CV_32FC1, CENTER_LIKELIHOOD).clone();
	delete[] CENTER_LIKELIHOOD;
}
// Detect object in the current frame.
cv::Point2f newTracker::detect(cv::Mat image, cv::Mat z, cv::Mat x, float &peak_value)
{
	using namespace FFTTools;

	cv::Mat k = gaussianCorrelation(x, z);
	cv::Mat res_cf = (real(fftd(complexMultiplication(_alphaf, fftd(k)), true)));
	
	//根据原来的bg，norm，放大一圈再缩小一圈至res_cf，为了结合hist模型，需要crop
	cv::Size newsz = _norm_delta_area;
	newsz.width = floor(newsz.width / cell_size);//定义searchSize
	newsz.height = floor(newsz.height / cell_size);

	(newsz.width % 2 == 0) && (newsz.width -= 1);//奇数
	(newsz.height % 2 == 0) && (newsz.height -= 1);

	cv::Mat response_cf;

	cropFilterResponse(res_cf, newsz, response_cf);
	if (cell_size > 1) {
		cv::Mat temp;
		mexResize(response_cf, temp, _norm_delta_area, "auto");
		response_cf = temp;
	}
	
	cv::Mat response_pwp;
	//cv::Mat response_pwp_temp;
	if (_colorhistfeatures) {
		cv::Size pwp_search_area;

		pwp_search_area.width = round(_norm_pwp_search_area.width * _area_resize_factor);
		pwp_search_area.height = round(_norm_pwp_search_area.height * _area_resize_factor);
		

		//getSubwindow(image, _pos, _norm_pwp_search_area, pwp_search_area, _im_patch_pwp);
		getSubwindow(image, _pos, _norm_pwp_search_area, pwp_search_area, _im_patch_pwp);

		cv::Mat likelihood_map;
		getColourMap(_im_patch_pwp, likelihood_map);

		//计算以当前点为中心的概率
		getCenterLikelihood(likelihood_map, _norm_target_sz, response_pwp);

		//mexResize(response_pwp_temp, response_pwp, res_cf.size(), "auto");
	}
	
	//minMaxLoc only accepts doubles for the peak, and integer points for the coordinates
	cv::Point2i pi;
	double pv;
	cv::Mat res;

	if (_hogfeatures&&_colorhistfeatures){
		cv::Point2i pi1, pi2;
		double pt1, pt2;
		cv::minMaxLoc(response_cf, NULL, &pt1, NULL, &pi1);
		cv::minMaxLoc(response_pwp, NULL, &pt2, NULL, &pi2);

		if ((pt1 > maxFHog*0.0) && (pt2 > maxColorHist*0.0)) {
			res = (1 - _alpha)*response_cf + _alpha * response_pwp;
		}else if(pt2 > maxColorHist*0.7){
			res = response_pwp;
		}
		else {
			res = response_cf;
		}
		
		maxFHog = maxFHog > pt1 ? maxFHog : pt1;
		maxColorHist = maxColorHist > pt2 ? maxColorHist : pt2;
		//
		//res = (1 - _alpha)*response_cf + _alpha * response_pwp;
	}
	else if (_hogfeatures)
		res = response_cf;
	else if (_colorhistfeatures)
		res = response_pwp;
	else
		res = 0;
	//测试峰值
	
	cv::Point2i pi1, pi2;
	double pt1, pt2;
	cv::minMaxLoc(response_cf, NULL, &pt1, NULL, &pi1);
	cv::minMaxLoc(response_pwp, NULL, &pt2, NULL, &pi2);
	if (testNumber % 3 == 1) {
		f << ((testNumber - 1) / 3 + 1) << "  :  " << pt1 << "---" << pt2 << endl;
	}
	testNumber++;
	
	//
	cv::minMaxLoc(res, NULL, &pv, NULL, &pi);
	peak_value = (float)pv;

	//subpixel peak estimation, coordinates will be non-integer
	cv::Point2f p((float)pi.x, (float)pi.y);

	/*原代码中的子像素峰值移动，删除
	if (pi.x > 0 && pi.x < res.cols - 1) {
		p.x += subPixelPeak(res.at<float>(pi.y, pi.x - 1), peak_value, res.at<float>(pi.y, pi.x + 1));
	}

	if (pi.y > 0 && pi.y < res.rows - 1) {
		p.y += subPixelPeak(res.at<float>(pi.y - 1, pi.x), peak_value, res.at<float>(pi.y + 1, pi.x));
	}
	*/
	p.x -= (res.cols) / 2;
	p.y -= (res.rows) / 2;

	return p;
}

void newTracker::cropFilterResponse(const cv::Mat &response_cf, cv::Size response_size, cv::Mat& output) {
	cv::Size sz = response_cf.size();
	int w = sz.width;
	int h = sz.height;

	// newh and neww must be odd, as we want an exact center
	assert(((response_size.width % 2) == 1) && ((response_size.height % 2) == 1));

	int half_width = floor(response_size.width / 2);
	int half_height = floor(response_size.height / 2);

	cv::Range i_range(-half_width, response_size.width - (1 + half_width));
	cv::Range j_range(-half_height, response_size.height - (1 + half_height));

	//cv::Range i_range(0, response_size.width - 1);
	//cv::Range j_range(0, response_size.height - 1);

	std::vector<int> i_mod_range, j_mod_range;
	int w_t = floor(w / 2);
	int h_t = floor(h / 2);
	//int temp;
	for (int k = i_range.start; k <= i_range.end; k++) {
		//int val = (k - 1 + w) % w;
		//temp = w_t + k;
		i_mod_range.push_back(w_t + k);
	}

	for (int k = j_range.start; k <= j_range.end; k++) {
		//int val = (k - 1 + h) % h;
		j_mod_range.push_back(h_t + k);
	}

	float *OUTPUT = new float[response_size.width*response_size.height];

	for (int i = 0; i < response_size.width; i++)
		for (int j = 0; j < response_size.height; j++) {
			int i_idx = i_mod_range[i];
			int j_idx = j_mod_range[j];

			assert((i_idx < w) && (j_idx < h));

			OUTPUT[j*response_size.width + i] = response_cf.at<float>(j_idx, i_idx);
		}

	output = cv::Mat(response_size.height, response_size.width, CV_32FC1, OUTPUT).clone();
	delete[] OUTPUT;
}
// Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y, which must both be MxN. They must    also be periodic (ie., pre-processed with a cosine window).
cv::Mat newTracker::gaussianCorrelation(cv::Mat x1, cv::Mat x2)
{
	using namespace FFTTools;
	cv::Mat c = cv::Mat(cv::Size(size_patch[1], size_patch[0]), CV_32F, cv::Scalar(0));
	// HOG features
	if (_hogfeatures) {
		cv::Mat caux;
		cv::Mat x1aux;
		cv::Mat x2aux;
		for (int i = 0; i < size_patch[2]; i++) {
			x1aux = x1.row(i);   // Procedure do deal with cv::Mat multichannel bug
			x1aux = x1aux.reshape(1, size_patch[0]);
			x2aux = x2.row(i).reshape(1, size_patch[0]);
			cv::mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true);
			caux = fftd(caux, true);
			rearrange(caux);
			caux.convertTo(caux, CV_32F);
			c = c + real(caux);
		}
	}
	// Gray features
	else {
		cv::mulSpectrums(fftd(x1), fftd(x2), c, 0, true);
		c = fftd(c, true);
		rearrange(c);
		c = real(c);
	}
	cv::Mat d;
	cv::max(((cv::sum(x1.mul(x1))[0] + cv::sum(x2.mul(x2))[0]) - 2. * c) / (size_patch[0] * size_patch[1] * size_patch[2]), 0, d);

	cv::Mat k;
	cv::exp((-d / (sigma * sigma)), k);
	return k;
	/*
	using namespace FFTTools;
	cv::Mat k;

	cv::Mat c = cv::Mat(cv::Size(size_patch[1], size_patch[0]), CV_32F, cv::Scalar(0));
	cv::Mat caux;
	cv::Mat x1aux;
	cv::Mat x2aux;
	for (int i = 0; i < size_patch[2]; i++) {
		x1aux = x1.row(i);   // Procedure do deal with cv::Mat multichannel bug
		x1aux = x1aux.reshape(1, size_patch[0]);
		x2aux = x2.row(i).reshape(1, size_patch[0]);
		cv::mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true);
		caux = fftd(caux, true);
		rearrange(caux);
		caux.convertTo(caux, CV_32F);
		c = c + real(caux);
	}
	
	cv::Mat d;
	cv::max(((cv::sum(x1.mul(x1))[0] + cv::sum(x2.mul(x2))[0]) - 2. * c) / (size_patch[0] * size_patch[1] * size_patch[2]), 0, d);

	cv::exp((-d / (sigma * sigma)), k);
	return k;
	*/
}

// Initialize Hanning window. Function called only in the first frame.
void newTracker::createHanningMats()
{
	cv::Mat hann1t = cv::Mat(cv::Size(size_patch[1], 1), CV_32F, cv::Scalar(0));
	cv::Mat hann2t = cv::Mat(cv::Size(1, size_patch[0]), CV_32F, cv::Scalar(0));

	for (int i = 0; i < hann1t.cols; i++)
		hann1t.at<float >(0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann1t.cols - 1)));
	for (int i = 0; i < hann2t.rows; i++)
		hann2t.at<float >(i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann2t.rows - 1)));

	/*
	cv::Mat hann2d = hann2t * hann1t;
	cv::Mat hann1d = hann2d.reshape(1, 1); // Procedure do deal with cv::Mat multichannel bug
	hann = cv::Mat(cv::Size(size_patch[0] * size_patch[1], size_patch[2]), CV_32F, cv::Scalar(0));
	for (int i = 0; i < size_patch[2]; i++) {
		for (int j = 0; j<size_patch[0] * size_patch[1]; j++) {
			hann.at<float>(i, j) = hann1d.at<float>(0, j);
		}
	}
	*/
	
	cv::Mat hann2d = hann2t * hann1t;
	// HOG features
	if (_hogfeatures) {
		cv::Mat hann1d = hann2d.reshape(1, 1); // Procedure do deal with cv::Mat multichannel bug

		hann = cv::Mat(cv::Size(size_patch[0] * size_patch[1], size_patch[2]), CV_32F, cv::Scalar(0));
		for (int i = 0; i < size_patch[2]; i++) {
			for (int j = 0; j<size_patch[0] * size_patch[1]; j++) {
				hann.at<float>(i, j) = hann1d.at<float>(0, j);
			}
		}
	}
	// Gray features
	else {
		hann = hann2d;
	}
	
}

// Create Gaussian Peak. Function called only in the first frame.
cv::Mat newTracker::createGaussianPeak(int sizey, int sizex)
{
	cv::Mat_<float> res(sizey, sizex);

	int syh = (sizey) / 2;
	int sxh = (sizex) / 2;

	float output_sigma = std::sqrt((float)sizex * sizey) / padding * output_sigma_factor;
	float mult = -0.5 / (output_sigma * output_sigma);

	for (int i = 0; i < sizey; i++)
		for (int j = 0; j < sizex; j++)
		{
			int ih = i - syh;
			int jh = j - sxh;
			res(i, j) = std::exp(mult * (float)(ih * ih + jh * jh));
		}
	return FFTTools::fftd(res);
}

// Calculate sub-pixel peak for one dimension
float newTracker::subPixelPeak(float left, float center, float right)
{
	float divisor = 2 * center - right - left;

	if (divisor == 0)
		return 0;

	return 0.5 * (right - left) / divisor;
}

//读YUVIamge
cv::Mat newTracker::readYUVImage(string imagePath, cv::Size imgYUVSize) {
	cv::Mat show_img;
	unsigned char* pYuvBuf = NULL;
	int bufLen = 0;

	FILE* pFileIn = fopen(imagePath.c_str(), "r");

	bufLen = imgYUVSize.width * imgYUVSize.height * 3 / 2;
	pYuvBuf = new unsigned char[bufLen]();
	fread(pYuvBuf, bufLen * sizeof(unsigned char), 1, pFileIn);
	cv::Mat imageTemp(imgYUVSize.height * 3 / 2, imgYUVSize.width, CV_8UC1, pYuvBuf);
	cvtColor(imageTemp, show_img, CV_YUV2BGR_NV21);
	fclose(pFileIn);

	return show_img;
}

//保存image
std::string newTracker::saveImage(string saveImgPath, bool hog, bool lab, bool colorhist, bool multiscale) {
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

	//std::string savePath = "D:\\1-documents\\1.1-ObjectTracking\\1.1.4-CaseTest\\TestResults\\";
	std::string filesName = month + day;
	if (multiscale)
		filesName += "-M";
	if (hog)
		filesName += "-H";
	if (lab)
		filesName += "-L";
	if (colorhist)
		filesName += "-C";
	saveImgPath += filesName;
	if (access(saveImgPath.c_str(), 0) == -1)
		int flag = mkdir(saveImgPath.c_str());

	return saveImgPath;
}


void newTracker::saveVideo(string saveImgPath, string saveVideoPath, string videoNameTemp,
	cv::Size imgSize, bool multiscale, bool hog, bool lab, bool colorhist) {
	//saveImgPath     e.g.     ..//TestResults//newTracker726-H-M//
	//saveVideoPath    e.g.    ..//TestResulrs//
	//videoNameTemp    e.g.    newTracker
	//imgSize          

	string videoName = "";

	videoName += videoNameTemp;

	std::time_t nowtime;
	nowtime = time(NULL);
	struct tm *local;
	local = localtime(&nowtime);
	ostringstream smonth, sday, shour, smin;
	std::string month, day, hour, min;
	
	smonth << (local->tm_mon + 1);
	month = smonth.str();
	sday << local->tm_mday;
	day = sday.str();
	shour << local->tm_hour;
	hour = shour.str();
	smin << local->tm_min;
	min = smin.str();

	//
	videoName = videoName + "-" + month + day + "-" + hour + min + "-";
	
	if (multiscale)
		videoName += "M";
	if(hog)
		videoName += "H";
	if(lab)
		videoName += "L";
	if (colorhist)
		videoName += "C";
	
	videoName = saveVideoPath + videoName;
	cv::VideoWriter video(videoName+".avi", CV_FOURCC('M', 'J', 'P', 'G'), 6.0, imgSize, 1);

	String pattern = saveImgPath + "\\*.jpg";
	vector<String> fn;

	glob(pattern, fn, false);
	size_t count = fn.size();
	for (size_t i = 0; i < count; i++)
	{
		Mat image = imread(fn[i]);
		
		resize(image, image, imgSize);
		
		video << image;
	}


}


