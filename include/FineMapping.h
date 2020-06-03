#ifndef SWIFTPR_FINEMAPPING_H
#define SWIFTPR_FINEMAPPING_H

#include <opencv2/opencv.hpp>
//#include <opencv2/dnn.hpp>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <string>
using namespace caffe;
namespace pr{
    class FineMapping{
    public:
        FineMapping();

        cv::Mat warpPerspect(cv::Mat src, int width,int height,int num);

        FineMapping(std::string prototxt,std::string caffemodel);
        void InitImage(const cv::Mat& img, std::vector<cv::Mat>* input_channels);
        cv::Mat FineMappingVertical(cv::Mat InputProposal,int sliceNum=15,int upper=0,int lower=-50,int windows_size=17);
        cv::Mat FineMappingHorizon(cv::Mat FinedVertical,int leftPadding,int rightPadding);


    private:
        // cv::dnn::Net net;
        std::shared_ptr<caffe::Net<float> > net_;
  		cv::Size input_geometry_;
  		int num_channels_;

    };




}
#endif //SWIFTPR_FINEMAPPING_H
