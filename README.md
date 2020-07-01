# TensorRT-LPR
车牌识别
该工程车牌检测技术参考YOLOv3，识别技术参考[HyperLPR](https://github.com/zeusees/HyperLPR)

工程基于[HyperLPR/prj-Linux](https://github.com/zeusees/HyperLPR/tree/master/Prj-Linux)修改：

1.修改车牌检测方法，改为yolov3实现；

2.源工程使用opencv-dnn模块对模型进行部署运行，改为caffe+tensorRT部署，可调用GPU进行加速

## 使用

### 工程环境：

 1. [caffe](https://github.com/BVLC/caffe)
 2. OpenCV 3.4.2
 3. TensorRT 6.0.1.5

### 依赖三方工程：

 1.[tensorRTWrapper](https://github.com/lewes6369/tensorRTWrapper):用于部署YOLOv3等模型；
 
 2.[darknet](https://github.com/pjreddie/darknet):YOLOv3车牌检测模型训练；
 
 3.[darkner2caffe](https://github.com/ChenYingpeng/darknet2caffe):darknet模型转为caffe模型。

### 模型:
1. yolov3对象检测模型[百度](https://pan.baidu.com/s/1ceroAl2aQCOwDwmUl80jAQ)(提取码：vm66 )
 
2. 车牌垂直边界回归模型
 
3. e2e车牌识别模型 

yolo对象检测模型可使用darknet自行训练，数据集参考[CCPD](https://github.com/detectRecog/CCPD)

部署时需要使用[darkner2caffe](https://github.com/ChenYingpeng/darknet2caffe)将darknet模型转换为caffe模型（注意工程中yolov3输入尺寸为418）

车牌边界回归以及识别模型来自[HyperLPR/prj-Linux](https://github.com/zeusees/HyperLPR/tree/master/Prj-Linux)

### 开始:
```Bash
git clone git@github.com:Fanghc95/TensorRT-LPR.git
#编辑CMakeList.txt配置CUDA，tensorRT，opencv等
mkdir build&&cd build
cmake ../
make -j8
./testPlate  [img_path]
```

### 改进方面:
1. 垂直边界回归没能用到tensorRT，我在部署时没能跑通，大佬们可以继续改进

2. 字符识别部分我用的是开源模型，效果较好但称不上100%完美，因为没接触过OCR没有进行改进（其实也莫得数据）

3. 公开数据集[CCPD](https://github.com/detectRecog/CCPD)虽数据量大但是场景单一，检测部分训练还需要额外数据进行优化
