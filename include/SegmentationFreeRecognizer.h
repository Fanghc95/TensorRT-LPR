#ifndef SWIFTPR_SEGMENTATIONFREERECOGNIZER_H
#define SWIFTPR_SEGMENTATIONFREERECOGNIZER_H

#include "Recognizer.h"
using namespace caffe;
namespace pr{

    unique_ptr<float[]> prepareImage(cv::Mat img);
    class SegmentationFreeRecognizer{
    public:
        const int CHAR_INPUT_W = 14;
        const int CHAR_INPUT_H = 30;
        const int CHAR_LEN = 84;

        SegmentationFreeRecognizer(std::string segregEngine);
        SegmentationFreeRecognizer(std::string prototxt, std::string caffemodel, const std::string& saveName);
        plateNum SegmentationFreeForSinglePlate(cv::Mat plate,std::vector<std::string> mapping_table);

    private:
        // std::vector<std::vector<float>> calibratorData;
        // trtNet net("model/SegmentationFree.prototxt","model/SegmentationFree.caffemodel",{"prob"},calibratorData);
        std::unique_ptr<trtNet> net;//RT网络框架
    };

}
#endif //SWIFTPR_SEGMENTATIONFREERECOGNIZER_H
