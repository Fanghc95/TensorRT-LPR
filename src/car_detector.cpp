#include "car_detector.h"

using namespace Yolo;
using namespace Tn;
using namespace std;
using namespace cv;

CarDetector::CarDetector(const std::string& prototxt, const std::string& caffemodel, const std::string& saveName){
	vector<vector<float>> calibData;
    //string saveName = "yolov3_car_fp32.engine";
    net.reset(new trtNet(prototxt, caffemodel, 80, {"yolo-det"}, calibData));
    cout << "save Engine..." << saveName <<endl;
    net->saveEngine(saveName);
    outputCount = net->getOutputSize()/sizeof(float);
}


CarDetector::CarDetector(const std::string& engineName){
	net.reset(new trtNet(engineName));
    outputCount = net->getOutputSize()/sizeof(float);
}

vector<float> CarDetector::prepareImage(cv::Mat& img)
{
    int c = 3;
    int h = 608;   //net h
    int w = 608;   //net w

    float scale = min(float(w)/img.cols,float(h)/img.rows);
    auto scaleSize = cv::Size(img.cols * scale,img.rows * scale);

    cv::Mat rgb ;
    cv::cvtColor(img, rgb, CV_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized, scaleSize, 0, 0, INTER_CUBIC);

    cv::Mat cropped(h, w, CV_8UC3, 127);
    // 按比例缩放 再居中放置
    Rect rect((w-scaleSize.width)/2, (h-scaleSize.height)/2, scaleSize.width,scaleSize.height); 
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


void CarDetector::DoNms(vector<Detection>& detections, float nmsThresh)
{
    int classes = 80;
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
//    cout << "Time taken for nms is " << total << " ms." << endl;
}


vector<Bbox> CarDetector::postProcessImg(cv::Mat& img, vector<Detection>& detections, float threshold)
{
    using namespace cv;

    int h = 608;   //net h
    int w = 608;   //net w

    //scale bbox to img
    int width = img.cols;
    int height = img.rows;
    float scale = min(float(w)/width,float(h)/height);
    float scaleSize[] = {width * scale,height * scale};

    //correct box
    //bbox[0], bbox[1]为中心点坐标x, y且在0-1之间
    //bbox[2], bbox[3]为bbox宽高, 之前未归一化, 在此进行归一化
    for (auto& item : detections)
    {
        auto& bbox = item.bbox;
        bbox[0] = (bbox[0] * w - (w - scaleSize[0])/2.f) / scaleSize[0];
        bbox[1] = (bbox[1] * h - (h - scaleSize[1])/2.f) / scaleSize[1];
        bbox[2] /= scaleSize[0];
        bbox[3] /= scaleSize[1];
    }

    //nms
    float nmsThresh = 0.45;
    DoNms(detections, nmsThresh);

    vector<Bbox> boxes;
    for(const auto& item : detections)
    {
        auto& b = item.bbox;
        //只要奔跑类别
        
        if((item.classId == 2 ||item.classId == 5 ||item.classId == 7) && item.prob > threshold){
            Bbox bbox = 
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
    }

    return boxes;
}

std::vector<Bbox> CarDetector::DetectCar(cv::Mat& img, const float threshold = 0.5){
    unique_ptr<float[]> outputData(new float[outputCount]);
    vector<float> curInput = prepareImage(img);

    //前向
    net->doInference(curInput.data(), outputData.get(), 1);

    //得到输出
    // list<vector<Bbox>> outputs;
    auto output = outputData.get();

    //first detect count
    int detCount = output[0];
    //later detect result
    vector<Detection> result;
    result.resize(detCount);
    memcpy(result.data(), &output[1], detCount*sizeof(Detection));

    auto boxes = postProcessImg(img, result, threshold);

    // net->printTime();

    outputData.reset(nullptr);
    return boxes;
}

