#include "Pipeline.h"
#include <time.h>
#include <string.h>
using namespace std;

void TEST_PIPELINE(char* path){//车牌识别
	pr::PipelinePR *prc;
    prc =new pr::PipelinePR("../model/yolov3_Plate.engine",
                    "../model/HorizonalFinemapping.prototxt","../model/HorizonalFinemapping.caffemodel",
                    "../model/SegmentationFree.engine"
                    );
    cv::Mat image = cv::imread(path);
    std::vector<pr::PlateInfo> res;
    res = prc->RunPiplineAsImage(image);
    for(auto st:res) {
        if(st.confidence>0.70) {
            std::cout<<"车牌号："<< st.getPlateName()<<endl;
            std::cout<<"置信度："<< st.getConfidence() <<endl;
            cout<<"各字符置信度："<<endl;
            for(int count=0; count<st.length;count++){
                cout<<st.nameList[count]<<':';
                cout<<st.confList[count]<<'\n';
            }
            cout<<endl;
        }
    }
    
}

void build_detection_engine()
{
    pr::PlateDetection *pp=new pr::PlateDetection("../model/yoloout.prototxt","../model/yoloout.caffemodel","../model/yolov3_Plate.engine");
}

void build_ocr_engine()
{
    pr::SegmentationFreeRecognizer *ocr=new pr::SegmentationFreeRecognizer("../model/SegmentationFree.prototxt","../model/SegmentationFree.caffemodel","../model/SegmentationFree.engine");
}


int main(int argc, char** argv)
{
	if(argc==2)
    {
        char mode[3];
	    strcpy(mode , argv[1]);
        if(mode[0]=='0')
        {
            cout<<"构建车牌检测engine"<<endl;
            build_detection_engine();
        }
        else
        {
            cout<<"构建车牌识别engine"<<endl;
            build_ocr_engine();
        }
    }
	else
	{
        char path[80];
		cout<<"部署环境并测试"<<endl;
		strcpy(path , argv[2]);
		TEST_PIPELINE(path);
	}
    return 0 ;
}
