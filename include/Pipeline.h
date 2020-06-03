#ifndef SWIFTPR_PIPLINE_H
#define SWIFTPR_PIPLINE_H

#include "PlateDetection.h"
#include "PlateInfo.h"
#include "FastDeskew.h"
#include "FineMapping.h"
#include "SegmentationFreeRecognizer.h"
char *getStrNum(int i, char num[3]);

namespace pr{

    const std::vector<std::string> CH_PLATE_CODE{"京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
                                        "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
                                        "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
                                        "Y", "Z","港","学","使","警","澳","挂","军","北","南","广","沈","兰","成","济","海","民","航","空"};



    const int SEGMENTATION_FREE_METHOD = 0;
    const int SEGMENTATION_BASED_METHOD = 1;

    class PipelinePR{
        public:
            // GeneralRecognizer *generalRecognizer;
            PlateDetection *plateDetection;
            // PlateSegmentation *plateSegmentation;
            FineMapping *fineMapping;
            SegmentationFreeRecognizer *segmentationFreeRecognizer;
            // PlateType *platetype; 

            PipelinePR(std::string detector_engin,
                        std::string finemapping_prototxt, std::string finemapping_caffemodel,
                        std::string segregEngine
                        );
            PipelinePR(std::string detect_prototxt,std::string detect_caffemodel,const std::string& saveName1,
                        std::string finemapping_prototxt, std::string finemapping_caffemodel,
                        std::string seg_prototxt,std::string seg_caffemodel, const std::string& saveName2
                        );
            ~PipelinePR();

            std::vector<std::string> plateRes;
            std::vector<PlateInfo> RunPiplineAsImage(cv::Mat plateImage,int method, int tool = 0);
    };


}
#endif //SWIFTPR_PIPLINE_H
