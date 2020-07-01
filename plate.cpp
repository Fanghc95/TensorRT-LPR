#include "Pipeline.h"
#include <time.h>
#include <string.h>
#include <stdio.h>
#include <io.h>
using namespace std;
int file_exists(char *filename)
{
    return (access(filename, 0) == 0);
}

void TEST_PIPELINE(char* path){//车牌识别
    pr::PipelinePR *prc;
    if(file_exists("yolov3_Plate.engine"))
    {
        prc =new pr::PipelinePR("../model/yolov3_Plate.engine",
                    "../model/HorizonalFinemapping.prototxt","../model/HorizonalFinemapping.caffemodel",
                    "../model/SegmentationFree.engine"
                    );
    }
    else
    {
        prc =new pr::PipelinePR prc("../model/yolov3_Plate.prototxt","../model/yolov3_Plate.caffemodel",
                    "../model/HorizonalFinemapping.prototxt","../model/HorizonalFinemapping.caffemodel",
                    "../model/SegmentationFree.prototxt","../model/SegmentationFree.caffemodel"
                    );
    }
    
    

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






int main(int argc, char** argv)
{
	char path[80];
	strcpy(path , argv[1]);

    TEST_PIPELINE(path);
   
    return 0 ;
}
