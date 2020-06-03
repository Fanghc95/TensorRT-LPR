#ifndef CAR_DETECTOR_H
#define CAR_DETECTOR_H

#include <opencv2/opencv.hpp>
#include "TrtNet.h"
#include <vector>
#include <chrono>
#include "YoloLayer.h"


class CarDetector
{
public:
	//Load from caffe model
	CarDetector(const std::string& prototxt, const std::string& caffemodel, const std::string& saveName);
	//Load from engine file
	explicit CarDetector(const std::string& engineName);
	std::vector<Yolo::Bbox> DetectCar(cv::Mat& img,const float threshold);
private:
	std::unique_ptr<Tn::trtNet> net;
	int outputCount;
	std::vector<float> prepareImage(cv::Mat& img);
	std::vector<Yolo::Bbox> postProcessImg(cv::Mat& img, std::vector<Yolo::Detection>& detections, float threshold);
	void DoNms(std::vector<Yolo::Detection>& detections, float nmsThresh);
};
#endif
