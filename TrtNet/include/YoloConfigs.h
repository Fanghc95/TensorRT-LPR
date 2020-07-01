#ifndef _YOLO_CONFIGS_H_
#define _YOLO_CONFIGS_H_


namespace Yolo
{
    static constexpr int CHECK_COUNT = 3;
    static constexpr float IGNORE_THRESH = 0.5f;
    // static constexpr int CLASS_NUM = 2; //No more need

    struct YoloKernel
    {
        int width;
        int height;
        float anchors[CHECK_COUNT*2];
    };

    //YOLO 608
    

    //YOLO 416
    
}

#endif
