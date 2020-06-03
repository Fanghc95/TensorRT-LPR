#include "../include/FineMapping.h"
namespace pr{

    const int FINEMAPPING_H = 60 ;
    const int FINEMAPPING_W = 140;
    const int PADDING_UP_DOWN = 30;
    void drawRect(cv::Mat image,cv::Rect rect)
    {
        cv::Point p1(rect.x,rect.y);
        cv::Point p2(rect.x+rect.width,rect.y+rect.height);
        cv::rectangle(image,p1,p2,cv::Scalar(0,255,0),1);
    }
    cv::Mat FineMapping::warpPerspect(cv::Mat src, int width,int height,int num)
    {
        int x1=0;
        int y1=0;
        int x2=0;
        int y2=0;
        int x3=0;
        int y3=0;
        int x4=0;
        int y4=0;
        int xc=0;
        int yc=0;


        int srcWid = src.cols;
        int srcHgh = src.rows;
        int midX = int(srcWid/2);
        int midY = int(srcHgh/2);
       
        int tx1 = midX+2*(x1-xc);
        int ty1 = midY+2*(y1-yc);
        int tx2 = midX+2*(x2-xc);
        int ty2 = midY+2*(y2-yc);
        int tx3 = midX+2*(x3-xc);
        int ty3 = midY+2*(y3-yc);
        int tx4 = midX+2*(x4-xc);
        int ty4 = midY+2*(y4-yc);

        std::vector<cv::Point2f> corners(4);
        corners[0] = cv::Point2f(tx1, ty1);
        corners[1] = cv::Point2f(tx2, ty2);
        corners[2] = cv::Point2f(tx3, ty3);
        corners[3] = cv::Point2f(tx4, ty4);
        std::vector<cv::Point2f> corners_trans(4);
        corners_trans[0] = cv::Point2f(440, 140);
        corners_trans[1] = cv::Point2f(0, 140);
        corners_trans[2] = cv::Point2f(440, 0);
        corners_trans[3] = cv::Point2f(0, 0);
        cv::Mat transform = cv::getPerspectiveTransform(corners, corners_trans);
	    cv::Mat quad = cv::Mat::zeros(140, 440, CV_8UC3);
        cv::warpPerspective(src, quad, transform, quad.size());
        return quad;
    }

    FineMapping::FineMapping(std::string prototxt,std::string caffemodel) {
        #ifdef CPU_ONLY
            Caffe::set_mode(Caffe::CPU);
	        std::cout<<"cpu"<<std::endl;
        #else
            Caffe::set_mode(Caffe::GPU);
	    std::cout<<"gpu"<<std::endl;
        #endif
        FLAGS_minloglevel = 3;
        net_.reset(new Net<float>(prototxt, TEST));
        net_->CopyTrainedLayersFrom(caffemodel);
        Blob<float>* input_layer = net_->input_blobs()[0];
        num_channels_ = input_layer->channels();
        input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

         //net = cv::dnn::readNetFromCaffe(prototxt, caffemodel);

    }

    cv::Mat FineMapping::FineMappingHorizon(cv::Mat FinedVertical,int leftPadding,int rightPadding)
    {

        Blob<float>* input_layer = net_->input_blobs()[0];
        input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
        /* Forward dimension change to all layers. */
        net_->Reshape();

        std::vector<cv::Mat> input_channels;
        InitImage(FinedVertical, &input_channels);
        net_->Forward();
        Blob<float>* result_blob = net_->output_blobs()[0];
        const float* result = result_blob->cpu_data();
        int front = static_cast<int>(result[0]*FinedVertical.cols);
        int back = static_cast<int>(result[1]*FinedVertical.cols);
        front -= leftPadding ;
        if(front<0) front = 0;
        back +=rightPadding;
        if(back>FinedVertical.cols-1) back=FinedVertical.cols - 1;
        cv::Mat cropped  = FinedVertical.colRange(front,back).clone();
        return  cropped;
    }
    std::pair<int,int> FitLineRansac(std::vector<cv::Point> pts,int zeroadd = 0 )
    {
        std::pair<int,int> res;
        if(pts.size()>2)
        {
            cv::Vec4f line;
            cv::fitLine(pts,line,cv::DIST_HUBER,0,0.01,0.01);
            float vx = line[0];
            float vy = line[1];
            float x = line[2];
            float y = line[3];
            int lefty = static_cast<int>((-x * vy / vx) + y);
            int righty = static_cast<int>(((136- x) * vy / vx) + y);
            res.first = lefty+PADDING_UP_DOWN+zeroadd;
            res.second = righty+PADDING_UP_DOWN+zeroadd;
            return res;
        }
        res.first = zeroadd;
        res.second = zeroadd;
        return res;
    }

    void FineMapping::InitImage(const cv::Mat& img, std::vector<cv::Mat>* input_channels) 
    {
        Blob<float>* input_layer = net_->input_blobs()[0];
        int width = input_layer->width();
        int height = input_layer->height();
        float* input_data = input_layer->mutable_cpu_data();

        for (int i = 0; i < input_layer->channels(); ++i) {
            cv::Mat channel(height, width, CV_32FC1, input_data);
            input_channels->push_back(channel);
            input_data += width * height;
        }

        cv::Mat sample_resized;
        if (img.size() != input_geometry_)
            cv::resize(img, sample_resized, input_geometry_);
        else
            sample_resized = img;

        cv::Mat sample_float;
        sample_resized.convertTo(sample_float, CV_32FC3);
        cv::Mat sample_normalized;
        //cv::subtract(sample_float, cv::Scalar(0, 0, 0), sample_normalized);
        sample_normalized = sample_float.mul(0.003922);
        cv::split(sample_normalized, *input_channels);
    }



    cv::Mat FineMapping::FineMappingVertical(cv::Mat InputProposal,int sliceNum,int upper,int lower,int windows_size){
        cv::Mat PreInputProposal;
        cv::Mat proposal;
        cv::resize(InputProposal,PreInputProposal,cv::Size(FINEMAPPING_W,FINEMAPPING_H));
        if(InputProposal.channels() == 3)
            cv::cvtColor(PreInputProposal,proposal,cv::COLOR_BGR2GRAY);
        else
            PreInputProposal.copyTo(proposal);
        // this will improve some sen
        cv::Mat kernal = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(1,3));
        float diff = static_cast<float>(upper-lower);
        diff/=static_cast<float>(sliceNum-1);
        cv::Mat binary_adaptive;
        std::vector<cv::Point> line_upper;
        std::vector<cv::Point> line_lower;
        int contours_nums=0;
        for(int i = 0 ; i < sliceNum ; i++)
        {
            std::vector<std::vector<cv::Point> > contours;
            float k =lower + i*diff;
            cv::adaptiveThreshold(proposal,binary_adaptive,255,cv::ADAPTIVE_THRESH_MEAN_C,cv::THRESH_BINARY,windows_size,k);
            cv::Mat draw;
            binary_adaptive.copyTo(draw);
            cv::findContours(binary_adaptive,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
            for(auto contour: contours)
            {
                cv::Rect bdbox =cv::boundingRect(contour);
                float lwRatio = bdbox.height/static_cast<float>(bdbox.width);
                int  bdboxAera = bdbox.width*bdbox.height;
                if ((   lwRatio>0.7&&bdbox.width*bdbox.height>100 && bdboxAera<300)
                    || (lwRatio>3.0 && bdboxAera<100 && bdboxAera>10))
                {
                    cv::Point p1(bdbox.x, bdbox.y);
                    cv::Point p2(bdbox.x + bdbox.width, bdbox.y + bdbox.height);
                    line_upper.push_back(p1);
                    line_lower.push_back(p2);
                    contours_nums+=1;
                }
            }
        }
        if(contours_nums<41)
        {
            cv::bitwise_not(InputProposal,InputProposal);
            cv::Mat kernal = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(1,5));
            cv::Mat bak;
            cv::resize(InputProposal,bak,cv::Size(FINEMAPPING_W,FINEMAPPING_H));
            cv::erode(bak,bak,kernal);
            if(InputProposal.channels() == 3)
                cv::cvtColor(bak,proposal,cv::COLOR_BGR2GRAY);
            else
                proposal = bak;
            int contours_nums=0;
            for(int i = 0 ; i < sliceNum ; i++)
            {
                std::vector<std::vector<cv::Point> > contours;
                float k =lower + i*diff;
                cv::adaptiveThreshold(proposal,binary_adaptive,255,cv::ADAPTIVE_THRESH_MEAN_C,cv::THRESH_BINARY,windows_size,k);
                cv::Mat draw;
                binary_adaptive.copyTo(draw);
                cv::findContours(binary_adaptive,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
                for(auto contour: contours)
                {
                    cv::Rect bdbox =cv::boundingRect(contour);
                    float lwRatio = bdbox.height/static_cast<float>(bdbox.width);
                    int  bdboxAera = bdbox.width*bdbox.height;
                    if ((   lwRatio>0.7&&bdbox.width*bdbox.height>120 && bdboxAera<300)
                        || (lwRatio>3.0 && bdboxAera<100 && bdboxAera>10))
                    {

                        cv::Point p1(bdbox.x, bdbox.y);
                        cv::Point p2(bdbox.x + bdbox.width, bdbox.y + bdbox.height);
                        line_upper.push_back(p1);
                        line_lower.push_back(p2);
                        contours_nums+=1;
                    }
                }
            }
        }
            cv::Mat rgb;
            cv::copyMakeBorder(PreInputProposal, rgb, PADDING_UP_DOWN, PADDING_UP_DOWN, 0, 0, cv::BORDER_REPLICATE);
            std::pair<int, int> A;
            std::pair<int, int> B;
            A = FitLineRansac(line_upper, -4);
            B = FitLineRansac(line_lower, 4);
            int leftyB = A.first;
            int rightyB = A.second;
            int leftyA = B.first;
            int rightyA = B.second;
            int cols = rgb.cols;
            int rows = rgb.rows;
            std::vector<cv::Point2f> corners(4);
            corners[0] = cv::Point2f(cols - 1, rightyA);
            corners[1] = cv::Point2f(0, leftyA);
            corners[2] = cv::Point2f(cols - 1, rightyB);
            corners[3] = cv::Point2f(0, leftyB);
            std::vector<cv::Point2f> corners_trans(4);
            corners_trans[0] = cv::Point2f(136, 36);
            corners_trans[1] = cv::Point2f(0, 36);
            corners_trans[2] = cv::Point2f(136, 0);
            corners_trans[3] = cv::Point2f(0, 0);
            cv::Mat transform = cv::getPerspectiveTransform(corners, corners_trans);
            cv::Mat quad = cv::Mat::zeros(36, 136, CV_8UC3);
            cv::warpPerspective(rgb, quad, transform, quad.size());
        return quad;
    }
}


