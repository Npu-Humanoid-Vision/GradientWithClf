#ifndef GRAD_VISION_H
#define GRAD_VISION_H

// 调参开关
#define ADJUST_PARAMETER


// 正负样本的 lable
#define POS_LABLE 1
#define NEG_LABLE 0


#include <opencv2/opencv.hpp>
#include <fstream> 
#include <iostream>
#include <iomanip>
using namespace std;
using namespace cv;

#ifdef ADJUST_PARAMETER

// showing image in debugging 
#define SHOW_IMAGE(win_name, imgName) \
    namedWindow(win_name, WINDOW_NORMAL); \
    imshow(win_name, imgName); \

class ImgProcResult{

public:
	ImgProcResult(){};
	~ImgProcResult(){};
	 virtual void operator=(ImgProcResult &res) = 0;
private:
protected:

};
class ImgProc{

public:
	ImgProc(){};
	~ImgProc(){};
	virtual void imageProcess(cv::Mat img, ImgProcResult *Result) =0;
private:
protected:
	ImgProcResult *res;

};
#else

#include "imgproc.h"
#define SHOW_IMAGE(imgName, yayaya) ;

#endif 


// adjust_parameter
class GradVisionResult : public ImgProcResult
{
public:
    bool valid_;
    cv::Rect bound_box_;
    cv::RotatedRect rotated_box_;
public:
    GradVisionResult() {
        valid_ = false;
    }

    // adjust_parameter    
    virtual void operator=(ImgProcResult &res) {
        GradVisionResult *tmp = dynamic_cast<GradVisionResult *>(&res);
        valid_ = tmp->valid_;
        bound_box_ = tmp->bound_box_;
        rotated_box_ = tmp->rotated_box_;
    }

    void operator=(GradVisionResult &res) {
        valid_ = res.valid_;
        bound_box_ = res.bound_box_;
        rotated_box_ = res.rotated_box_;
    }
};

struct AllParameters_grad {
    int gaus_size;
    int channel_idx;
    int grad_thre;
    int area_thre;
    double wh_rate_thre;
};

class GradVision : public ImgProc {
public:
    GradVision();
    GradVision(string config_path);// 请给一个绝对路径，谢谢
    ~GradVision();

public: // 假装是接口的函数
    void imageProcess(cv::Mat input_image, ImgProcResult* output_result);   // 对外接口

    cv::Mat Pretreat(cv::Mat raw_image);                                    // 所有图像进行目标定位前的预处理

    cv::Mat ProcessGrad();                                                  // 梯度操作

    void GetPossibleRect(cv::Mat binary_image, 
                        std::vector<cv::Rect>& result_rects,
                        std::vector<cv::RotatedRect>& result_rrects);       // 获得待选区域

    cv::Mat GetFeatureVec(cv::Rect roi);                                    // roi特征工程

    cv::Mat GetUsedChannel(cv::Mat& src_img, int flag);                     // 获得使用的通道

public: // 真实的接口函数
    void LoadParameters(string config_path);                                // 从文件加载参数(注意为绝对路径)

    void StoreParameters();                                                 // 将参数存到文件

    void set_all_parameters(AllParameters_grad);                            // 调参时候传入参数

    void WriteImg(cv::Mat src, string folder_name, int num);                // 写图片

public: // 数据成员
    cv::Mat src_image_;
    cv::Mat pretreaded_image_;
    cv::Mat used_channel_;
    cv::Mat thresholded_image_;
    CvSVM svm_classifier_;
    GradVisionResult final_result_;

    // 参数
    int gaus_size_;
    int channel_idx_;
    int grad_thre_;
    int area_thre_;
    double wh_rate_thre_;

    // 存图相关
    int start_file_num_;
    int max_file_num_;

    // SVM model path
    string svm_model_name_;

    // 结果Rect
    cv::Rect result_rect_;
    cv::RotatedRect result_ro_rect_;
};



#endif