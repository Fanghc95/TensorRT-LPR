#ifndef SWIFTPR_FASTDESKEW_H
#define SWIFTPR_FASTDESKEW_H

#include <math.h>
#include <opencv2/opencv.hpp>
namespace pr{

    cv::Mat fastdeskew(cv::Mat skewImage,int blockSize);
//    cv::Mat spatialTransformer(cv::Mat skewImage);

}//namepace pr


#endif //SWIFTPR_FASTDESKEW_H
