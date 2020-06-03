#ifndef SWIFTPR_PLATEDETECTION_H
#define SWIFTPR_PLATEDETECTION_H

#include <opencv2/opencv.hpp>
//#include <PlateInfo.h>
#include "PlateInfo.h"
#include <vector>

#include "TrtNet.h"
#include <chrono>
#include "YoloLayer.h"
#include "dataReader.h"

using namespace std;
using namespace cv;
using namespace Tn;
using namespace Yolo;
namespace pr{
    
    class PlateDetection{
    public:
        vector<float> prepareImage(cv::Mat& img);
        vector<Yolo::Bbox> postProcessImg(cv::Mat& img,vector<Detection>& detections,int classes,float nmsThresh);
        void DoNms(vector<Detection>& detections,int classes,float nmsThresh);
        PlateDetection(std::string model_engine);
        PlateDetection(std::string prototxt, std::string caffemodel, const std::string &saveName);
        PlateDetection();
        void LoadModel(std::string model_engine);
        void plateDetectionRough(cv::Mat img,std::vector<pr::PlateInfo>  &plateInfos,int min_w=36,int max_w=800);

    private:
        std::unique_ptr<trtNet> net;//RT网络框架
        int outputCount;//网络输出尺寸
        int classNum;//类别数，1
        int c;//输入通道式
        int h;//输入宽高
        int w;
        float nmsThresh;

        vector<float> inputData;
        // unique_ptr<float[]> outputData;
    };

}// namespace pr

#endif //SWIFTPR_PLATEDETECTION_H
