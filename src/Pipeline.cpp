#include "../include/Pipeline.h"

char *getStrNum(int i, char num[4]) {
    char cu;
    //char num[3]="22";
    if (i >= 0 && i <= 9) {
        cu = '0' + i;
        //char res[2];
        num[0] = cu;
        num[1] = '\0';
        //strcpy(num, res);
    } else if (i >= 10 && i <= 99) {
        int d1 = int(i % 10);
        int d2 = int(i / 10);
        char cu1 = '0' + d1;
        char cu2 = '0' + d2;
        //char res[3];
        num[0] = cu2;
        num[1] = cu1;
        num[2] = '\0';
        //strcpy(num,res);
    } else {
        int d1 = int(i % 10);
        // int d2=int(i%100);
        int d3 = int(i / 100);
        int d2 = (i - d3 * 100) / 10;
        char cu1 = '0' + d1;
        char cu2 = '0' + d2;
        char cu3 = '0' + d3;
        //char res[3];
        num[0] = cu3;
        num[1] = cu2;
        num[2] = cu1;
        num[3] = '\0';
    }
    return num;
}

namespace pr {
    const int HorizontalPadding = 4;

    PipelinePR::PipelinePR(std::string detector_engin,
                           std::string finemapping_prototxt, std::string finemapping_caffemodel,
                           std::string segregEngine
    ) {
        plateDetection = new PlateDetection(detector_engin);
        fineMapping = new FineMapping(finemapping_prototxt, finemapping_caffemodel);
        segmentationFreeRecognizer = new SegmentationFreeRecognizer(segregEngine);
    }

    PipelinePR::PipelinePR(std::string detect_prototxt, std::string detect_caffemodel, const std::string &saveName1,
                           std::string finemapping_prototxt, std::string finemapping_caffemodel,
                           std::string seg_prototxt, std::string seg_caffemodel, const std::string &saveName2) {
        plateDetection = new PlateDetection(detect_prototxt, detect_caffemodel, saveName1);
        fineMapping = new FineMapping(finemapping_prototxt, finemapping_caffemodel);
        segmentationFreeRecognizer = new SegmentationFreeRecognizer(seg_prototxt, seg_caffemodel, saveName2);
    }

    PipelinePR::~PipelinePR() {
        delete plateDetection;
        delete fineMapping;
        delete segmentationFreeRecognizer;
    }

    std::vector<PlateInfo> PipelinePR::RunPiplineAsImage(cv::Mat plateImage, int method, int tool) {
        std::vector<PlateInfo> results;
        std::vector<pr::PlateInfo> plates;
        cv::Mat useDet = plateImage;
        plateDetection->plateDetectionRough(useDet, plates, 36, 700);
        int qqlq = 0;
        for (pr::PlateInfo plateinfo:plates) {
            qqlq++;
            cv::Mat image_finemapping = plateinfo.getPlateImage();
            float confid = 0.0;
            cv::Mat vRefineImg = fineMapping->FineMappingVertical(image_finemapping);

            vRefineImg = pr::fastdeskew(vRefineImg, 5);

            std::vector<cv::Mat> hRefineImg;

            cv::Mat resH = fineMapping->FineMappingHorizon(vRefineImg, 3, 4);
            cv::resize(resH, resH, cv::Size(136 + HorizontalPadding, 36));
            pr::plateNum res = segmentationFreeRecognizer->SegmentationFreeForSinglePlate(resH, pr::CH_PLATE_CODE);

            if (res.confidence > confid) {
                confid = res.confidence;
                plateinfo.confidence = res.confidence;
                plateinfo.length = res.length;
                plateinfo.nameList = res.nameList;
                plateinfo.confList = res.confList;
                plateinfo.setPlateName(res.name);
                plateinfo.setPlateImage(resH);
            }
            results.push_back(plateinfo);
        }
        return results;
    }
}
        

