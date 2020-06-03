#include "../include/PlateDetection.h"
#include "../include/util.h"


namespace pr{
    vector<string> split(const string& str, char delim)
{
    stringstream ss(str);
    string token;
    vector<string> container;
    while (getline(ss, token, delim)) {
        container.push_back(token);
    }

    return container;
}
    vector<float> PlateDetection::prepareImage(cv::Mat& img)
    {
        using namespace cv;

        float scale = min(float(w)/img.cols,float(h)/img.rows);
        auto scaleSize = cv::Size(img.cols * scale,img.rows * scale);

        cv::Mat rgb ;
        cv::cvtColor(img, rgb, 4);
        cv::Mat resized;
        cv::resize(rgb, resized,scaleSize,0,0,INTER_CUBIC);

        cv::Mat cropped(h, w,CV_8UC3, 127);
        Rect rect((w- scaleSize.width)/2, (h-scaleSize.height)/2, scaleSize.width,scaleSize.height); 
        resized.copyTo(cropped(rect));

        cv::Mat img_float;
        if (c == 3)
            cropped.convertTo(img_float, CV_32FC3, 1/255.0);
        else
            cropped.convertTo(img_float, CV_32FC1 ,1/255.0);

        //HWC TO CHW
        vector<Mat> input_channels(c);
        cv::split(img_float, input_channels);

        vector<float> result(h*w*c);
        auto data = result.data();
        int channelLength = h * w;
        for (int i = 0; i < c; ++i) {
            memcpy(data,input_channels[i].data,channelLength*sizeof(float));
            data += channelLength;
        }

        return result;
    }

    vector<Yolo::Bbox> PlateDetection::postProcessImg(cv::Mat& img,vector<Detection>& detections,int classes,float nmsThresh)
    {
        using namespace cv;

        //scale bbox to img
        int width = img.cols;
        int height = img.rows;
        float scale = min(float(w)/width,float(h)/height);
        float scaleSize[] = {width * scale,height * scale};

        //correct box
        for (auto& item : detections)
        {
            auto& bbox = item.bbox;
            bbox[0] = (bbox[0] * w - (w - scaleSize[0])/2.f) / scaleSize[0];
            bbox[1] = (bbox[1] * h - (h - scaleSize[1])/2.f) / scaleSize[1];
            bbox[2] /= scaleSize[0];
            bbox[3] /= scaleSize[1];
        }
        
        if(nmsThresh > 0) 
            DoNms(detections,classes,nmsThresh);

        vector<Yolo::Bbox> boxes;
        for(const auto& item : detections)
        {
            auto& b = item.bbox;
            Yolo::Bbox bbox =
            { 
                item.classId,   //classId
                max(int((b[0]-b[2]/2.)*width),0), //left
                min(int((b[0]+b[2]/2.)*width),width), //right
                max(int((b[1]-b[3]/2.)*height),0), //top
                min(int((b[1]+b[3]/2.)*height),height), //bot
                item.prob       //score
            };
            boxes.push_back(bbox);
        }

        return boxes;
    }

    void PlateDetection::DoNms(vector<Detection>& detections,int classes ,float nmsThresh)
    {
        auto t_start = chrono::high_resolution_clock::now();

        vector<vector<Detection>> resClass;
        resClass.resize(classes);

        for (const auto& item : detections)
            resClass[item.classId].push_back(item);

        auto iouCompute = [](float * lbox, float* rbox)
        {
            float interBox[] = {
                max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
                min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
                max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
                min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
            };
            
            if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
                return 0.0f;

            float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
            return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
        };

        vector<Detection> result;
        for (int i = 0;i<classes;++i)
        {
            auto& dets =resClass[i]; 
            if(dets.size() == 0)
                continue;

            sort(dets.begin(),dets.end(),[=](const Detection& left,const Detection& right){
                return left.prob > right.prob;
            });

            for (unsigned int m = 0;m < dets.size() ; ++m)
            {
                auto& item = dets[m];
                result.push_back(item);
                for(unsigned int n = m + 1;n < dets.size() ; ++n)
                {
                    if (iouCompute(item.bbox,dets[n].bbox) > nmsThresh)
                    {
                        dets.erase(dets.begin()+n);
                        --n;
                    }
                }
            }
        }

        //swap(detections,result);
        detections = move(result);

        auto t_end = chrono::high_resolution_clock::now();
        float total = chrono::duration<float, milli>(t_end - t_start).count();
//        cout << "Time taken for nms is " << total << " ms." << endl;
    }
    PlateDetection::PlateDetection(std::string prototxt, std::string caffemodel, const std::string &saveName){
        string deployFile = prototxt;
        string caffemodelFile = caffemodel;

        vector<vector<float>> calibData;
        string calibFileList = "";
        string mode = "fp32";
        
        RUN_MODE run_mode = RUN_MODE::FLOAT32;
        
        string outputNodes = "yolo-det";
        auto outputNames = split(outputNodes,',');
        
        //save Engine name
        //string saveName = "../model/yolov3_Plate.engine";
        net.reset(new trtNet(deployFile,caffemodelFile,1,outputNames,calibData, 416, run_mode,1));
        cout << "save Engine..." << saveName <<endl;
        net->saveEngine(saveName);
        assert(net->getBatchSize() == 1);
        outputCount = net->getOutputSize()/sizeof(float);

        classNum = 1;
        c = 3;
        h = 416;
        w = 416;
        nmsThresh = 0.5;
    }
    PlateDetection::PlateDetection(std::string model_engine){
        net.reset(new trtNet(model_engine));
        assert(net->getBatchSize() == 1);
        outputCount = net->getOutputSize()/sizeof(float);
        classNum = 1;
        c = 3;
        h = 416;
        w = 416;
        nmsThresh = 0.5;
    }
    void PlateDetection::plateDetectionRough(cv::Mat img,std::vector<pr::PlateInfo>  &plateInfos,int min_w,int max_w){

        vector<float> curInput = prepareImage(img);//预处理图片
        unique_ptr<float[]> outputData(new float[outputCount]);
        net->doInference(curInput.data(), outputData.get(), 1);

        //得到输出
        list<vector<Yolo::Bbox>> outputs;
        auto output = outputData.get();

        //first detect count
        int detCount = output[0];
        if (detCount == 0){
            return ;
        }
        //later detect result
        vector<Detection> result;
        result.resize(detCount);
        memcpy(result.data(), &output[1], detCount*sizeof(Detection));
        auto boxes = postProcessImg(img,result,classNum,nmsThresh);
        outputs.emplace_back(boxes);

        //draw on image
        auto bbox = *outputs.begin();
        for(const auto& item : bbox)
        {
            //反归一化，得到图像坐标
		    int xLeftUp = item.left;
		    int yLeftUp = item.top;

		    int xRightBottom = item.right;
		    int yRightBottom = item.bot;

            int w = xRightBottom-xLeftUp;
            int h = yRightBottom-yLeftUp;

            int nx = xLeftUp -0.2*w;
            int ny = yLeftUp-0.15*h;
            int mx = xRightBottom+0.2*w;
            int my = yRightBottom+0.15*h;
		    //矩形框
		    Rect rect(Point{ nx,ny }, Point{ mx,my });
            cv::Mat plateImage = util::cropFromImage(img,rect);
            PlateInfo plateInfo(plateImage,rect);
		    //保存结果
            plateInfos.push_back(plateInfo);
        }
    }
}//namespace pr
