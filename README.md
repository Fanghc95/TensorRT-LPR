# TensorRT-LPR
车牌识别
该工程车牌检测技术参考YOLOv3，识别技术参考[HyperLPR](https://github.com/zeusees/HyperLPR)

工程基于[HyperLPR/prj-Linux](https://github.com/zeusees/HyperLPR/tree/master/Prj-Linux)修改：

1.修改车牌检测方法，改为yolov3实现；

2.源工程使用opencv-dnn模块对模型进行部署运行，改为caffe+tensorRT部署，可调用GPU进行加速

3. ....

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
1. yolo对象检测模型
 
2. 车牌垂直边界回归模型
 
3. e2e车牌识别模型 

对象检测模型可使用darknet自行训练，再将darknet模型转换为caffe模型即可（工程中yolov3输入尺寸为418），其他模型来自HyperLPR

### 开始:
'''
git clone 

#编辑CMakeList.txt配置CUDA，tensorRT，opencv等
mkdir build&&cd build
cmake ../
make -j8
'''
