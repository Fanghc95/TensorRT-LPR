#ifndef SWIFTPR_OBJECTINFO_H
#define SWIFTPR_OBJECTINFO_H
#include<iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

namespace pr {
    class Object
    {
    public:
        Object(){}
        Object(int index, float confidence, String name, Rect rect)
        {
            this->index = index;
            this->confidence = confidence;
            this->name = name;
            this->rect = rect;
        }
        ~Object(){}

    public:
        int index;
        String name;
        float confidence;
        Rect rect;

        Rect getRect(){
            return rect;
        }
        String getName(){
            return name;
        }
        float getConfidence(){
            return confidence;
        }

    private:
    };

}
#endif //SWIFTPR_OBJECTINFO_H