#ifndef SWIFTPR_RECOGNIZER_H
#define SWIFTPR_RECOGNIZER_H

#include "plateNum.h"
#include "PlateInfo.h"
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
#include <sstream>
#include <opencv2/opencv.hpp>

#include "TrtNet.h"

using namespace std;
using namespace Tn;
namespace pr{
    typedef cv::Mat label;
    class GeneralRecognizer{
        public:
            virtual label recognizeCharacter(cv::Mat character) = 0;
//            virtual cv::Mat SegmentationFreeForSinglePlate(cv::Mat plate) = 0;
            void SegmentBasedSequenceRecognition(PlateInfo &plateinfo);
            void SegmentationFreeSequenceRecognition(PlateInfo &plateInfo);

    };

}
#endif //SWIFTPR_RECOGNIZER_H
