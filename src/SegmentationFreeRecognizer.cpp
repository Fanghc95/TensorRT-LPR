#include "../include/SegmentationFreeRecognizer.h"

namespace pr {
    SegmentationFreeRecognizer::SegmentationFreeRecognizer(std::string prototxt, std::string caffemodel, const std::string& saveName) {
        std::vector<std::vector<float>> calibratorData;
        net.reset(new trtNet(prototxt,caffemodel,0,{"prob"},calibratorData));
        net->saveEngine(saveName);
    }
    SegmentationFreeRecognizer::SegmentationFreeRecognizer(std::string segregEngine) {   
        net.reset(new trtNet(segregEngine));   
    }
    unique_ptr<float[]> prepareImage(cv::Mat img)
    {
        using namespace cv;
        cv::transpose(img,img); 

        int c = 3;
        int h = 160;
        int w = 40;

        cv::Mat resized;
        cv::resize(img, resized, cv::Size(w, h));

        cv::Mat img_float;
        if (c == 3)
            resized.convertTo(img_float, CV_32FC3);
        else
            resized.convertTo(img_float, CV_32FC1);
        cv::Mat sample_normalized;
        cv::subtract(img_float, cv::Scalar(0, 0, 0), sample_normalized);

        // cv::Mat initIMG;
        img_float = sample_normalized.mul(0.003922);

        //HWC TO CHW
        cv::Mat input_channels[c];
        cv::split(img_float, input_channels);

        float * data = new float[h*w*c];
        auto result = data;
        int channelLength = h * w;
        for (int i = 0; i < c; ++i) {
            memcpy(data,input_channels[i].data,channelLength*sizeof(float));
            data += channelLength;
        }     

        return std::unique_ptr<float[]>(result);
    }
    inline int judgeCharRange(int id)
    {return id<31 || id>63;}

    plateNum decodeResults(const float* result,const int height,const int width,std::vector<std::string> mapping_table,float thres)
    {
        int sequencelength =20;
        int labellength = 84;
        float q[1680];
        for(int i=0;i<20;i++)
        {
            for(int j=0;j<84;j++)
            {
            q[i*84+j]=result[20*j+i];
            }
        }
	    std::string name = "";
        std::vector<int> seq(sequencelength);
        std::vector<std::pair<int,float>> seq_decode_res;
        for(int i = 0 ; i < sequencelength;  i++) {
            float *fstart = ((float *) q+ i * labellength );
            int id = std::max_element(fstart,fstart+labellength) - fstart;
            seq[i] =id;
        }

        float sum_confidence = 0;
        int plate_lenghth  = 0 ;
        for(int i = 0 ; i< sequencelength ; i++)
        {
            if(seq[i]!=labellength-1 && (i==0 || seq[i]!=seq[i-1]))
            {
                float *fstart = ((float *) q + i * labellength );
                float confidence = *(fstart+seq[i]);
                std::pair<int,float> pair_(seq[i],confidence);
                seq_decode_res.push_back(pair_);
            }
        }
        int  i = 0;
        std::vector<std::string> single_name;
        std::vector<float> single_confid;
        if (seq_decode_res.size()>1 && judgeCharRange(seq_decode_res[0].first) && judgeCharRange(seq_decode_res[1].first))
        {
            i=2;
            int c = seq_decode_res[0].second<seq_decode_res[1].second;
            name+=mapping_table[seq_decode_res[c].first];
            sum_confidence+=seq_decode_res[c].second;
            single_name.push_back(mapping_table[seq_decode_res[c].first]);
            single_confid.push_back(seq_decode_res[c].second);
            plate_lenghth++;
            std::cout<<seq_decode_res[c].second<<'\t';
        }

        for(; i < seq_decode_res.size();i++)
        {
            name+=mapping_table[seq_decode_res[i].first];
            sum_confidence +=seq_decode_res[i].second;
            single_name.push_back(mapping_table[seq_decode_res[i].first]);
            single_confid.push_back(seq_decode_res[i].second);
            plate_lenghth++;
        }
        
        plateNum returnRes;
        if(plate_lenghth==7){
            returnRes.nameList=single_name;
            returnRes.confList=single_confid;
            returnRes.length = plate_lenghth;
            returnRes.name =name ;
            returnRes.confidence = sum_confidence/plate_lenghth;
        }
        else
        {
            returnRes.confidence = 0.0;
        }
        
        
        return returnRes;
    }


   plateNum SegmentationFreeRecognizer::SegmentationFreeForSinglePlate(cv::Mat Image,std::vector<std::string> mapping_table) {
       
        
        auto inputData = prepareImage(Image);
        int outputCount = net->getOutputSize()/sizeof(float);
        std::unique_ptr<float[]> outputData(new float[outputCount]);

        net->doInference(inputData.get(), outputData.get());

        auto result = outputData.get();

        return decodeResults(result,0,0,mapping_table,0.00);
    }

}
