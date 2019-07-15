#include "GradVision.h"

GradVision::GradVision() {
    final_result_.valid_    = false;
    start_file_num_         = 0;
    max_file_num_           = 500;
}

GradVision::GradVision(string config_path) {
    final_result_.valid_    = false;
    start_file_num_         = 0;
    max_file_num_           = 500;
    this->LoadParameters(config_path);
}

GradVision::~GradVision() {}

void GradVision::imageProcess(cv::Mat input_image, ImgProcResult* output_result) {
    std::vector<cv::Rect> possible_rects;
    std::vector<cv::RotatedRect> possible_rrects;

    std::vector<cv::Rect> pos_rects;
    std::vector<cv::RotatedRect> pos_rrects;
    cv::Point2f ro_rect_points[4];
    SHOW_IMAGE("living", input_image);

    cv::Mat for_show_ = input_image.clone();
    src_image_ = input_image.clone();
    pretreaded_image_ = Pretreat(src_image_);
    thresholded_image_ = ProcessGrad();
    GetPossibleRect(thresholded_image_, possible_rects, possible_rrects);

    for (size_t i=0; i<possible_rects.size(); i++) {
        cv::Rect t_bound_box = possible_rects[i];
        cv::Mat roi = src_image_(t_bound_box).clone();
        cv::Mat feature_in_mat = GetFeatureVec(t_bound_box);

        int lable = (int)svm_classifier_.predict(feature_in_mat);

        if (lable == POS_LABLE) {
            pos_rects.push_back(possible_rects[i]);
            pos_rrects.push_back(possible_rrects[i]);

            possible_rrects[i].points(ro_rect_points);
            for (int i=0; i<4; i++) {
                cv::line(for_show_, ro_rect_points[i], ro_rect_points[(i+1)%4], cv::Scalar(255, 255, 0));
            }
            cv::rectangle(for_show_, t_bound_box, cv::Scalar(0, 255, 0), 2);

        }
        else {
            cv::rectangle(for_show_, t_bound_box, cv::Scalar(0, 0, 255), 2);
        }
    }

    if (pos_rects.size() >= 1) {
        final_result_.valid_ = true;
        int max_area = -1;
        int max_idx = -1;
        for (int i=0; i<pos_rects.size(); i++) {
            cv::Rect t_bound_box = pos_rects[i];
            if (t_bound_box.area() > max_area) {
                max_area = t_bound_box.area();
                max_idx = i;
            }
        }
        final_result_.rotated_box_ = pos_rrects[max_idx];
        final_result_.bound_box_ = pos_rects[max_idx];

        final_result_.rotated_box_.points(ro_rect_points);
        for (int i=0; i<4; i++) {
            cv::line(for_show_, ro_rect_points[i], ro_rect_points[(i+1)%4], cv::Scalar(255, 255, 0));
        }
        cv::rectangle(for_show_, final_result_.bound_box_, cv::Scalar(255, 255, 0));
    }
    else {
        final_result_.valid_ = false;
    }

    SHOW_IMAGE("result", for_show_);

    (*dynamic_cast<GradVisionResult*>(output_result)) = final_result_;

#ifndef ADJUST_PARAMETER
    this->WriteImg(src_image_,"src_img",start_file_num_);
    if (final_result_.valid_) {
        cv::rectangle(for_show_, result_rect_, cv::Scalar(0, 255, 255));
    }
    this->WriteImg(for_show_,"center_img",start_file_num_++);
#endif
}

cv::Mat GradVision::Pretreat(cv::Mat raw_image) {
    cv::Mat blured_image;
    cv::GaussianBlur(raw_image, blured_image, cv::Size(2*gaus_size_+1, 2*gaus_size_+1), 0, 0);

    used_channel_ = GetUsedChannel(blured_image, channel_idx_);
    SHOW_IMAGE("gaused image", blured_image);
    return blured_image;
}

cv::Mat GradVision::ProcessGrad() {
    cv::Mat grad_x;
    cv::Mat grad_y;

    SHOW_IMAGE("used_channel", used_channel_);
    cv::Sobel(used_channel_, grad_x, CV_16S, 1, 0);
    cv::Sobel(used_channel_, grad_y, CV_16S, 0, 1);
    cv::convertScaleAbs(grad_x, grad_x);
    cv::convertScaleAbs(grad_y, grad_y);

    cv::Mat grad = 0.5*grad_x+0.5*grad_y;
    
    cv::Mat grad_thre = grad>grad_thre_;

    SHOW_IMAGE("grad_thre", grad_thre);
    return grad_thre;
}

void GradVision::GetPossibleRect(cv::Mat binary_image,
                                std::vector<cv::Rect>& result_rects,
                                std::vector<cv::RotatedRect>& result_rrects) {
    int row = src_image_.rows;
    int col = src_image_.cols;

    std::vector<std::vector<cv::Point> > contours;
    std::vector<std::vector<cv::Point> > contours_poly;

    cv::Mat image_for_contours = binary_image.clone();
    cv::findContours(image_for_contours, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

    contours_poly.resize(contours.size());

    int max_inter_area = 0.0;
    int min_dist_idx = -1;
    
    for (unsigned int i = 0; i < contours.size(); i++) {
        cv::approxPolyDP(contours[i], contours_poly[i], 3, false);
        cv::Rect t_rect = cv::boundingRect(contours_poly[i]);
        cv::RotatedRect t_rrect = cv::minAreaRect(contours_poly[i]);
        double wh_rate = t_rect.width*1.0/t_rect.height;
 
        // area & shape thre
        // 若三角形只检测到单条
        // 下单条
        // 右单条
        // 左单条
        cv::Rect t_new_rect;
        if (t_rect.width*t_rect.width > area_thre_) {
            if (wh_rate > wh_rate_thre_) {
                t_new_rect.width = t_rect.width*5./4;
                t_new_rect.height = t_new_rect.width;
                t_new_rect.x = t_rect.x - t_rect.width/8;
                t_new_rect.y = t_rect.y + t_rect.height - t_rect.width*7./8;
            }
            else if (wh_rate < 1.0/wh_rate_thre_) {
                t_new_rect.height = t_rect.height*5./4;
                t_new_rect.width = t_new_rect.height;
                t_new_rect.y = t_rect.y - t_rect.height/8;
                t_new_rect.x = t_rect.x + t_rect.height - t_rect.height*7./8;
            }
            else {
                t_new_rect.height = max(t_rect.height, t_rect.width);
                t_new_rect.height = t_new_rect.height*5./4;
                t_new_rect.width = t_new_rect.height;
                t_new_rect.y = t_rect.y - t_new_rect.height/8;
                t_new_rect.x = t_rect.x - t_new_rect.height/8;
            }
        }
        if (0<t_new_rect.x && t_new_rect.x<col
        &&  0<t_new_rect.y && t_new_rect.y<row
        &&  t_new_rect.x + t_new_rect.width < col
        &&  t_new_rect.y + t_new_rect.width < row) {
            result_rects.push_back(t_new_rect);
            result_rrects.push_back(t_rrect);
        }

    }
    return;
}

cv::Mat GradVision::GetFeatureVec(cv::Rect roi) {
    cv::Mat roi_in_mat = src_image_(roi).clone();

    cv::resize(roi_in_mat, roi_in_mat, cv::Size(128, 128)); // 与训练相关参数，之后最好做成文件传入参数
    cv::HOGDescriptor hog_des(Size(128, 128), Size(16,16), Size(8,8), Size(8,8), 9);
    std::vector<float> feature_vec;
    hog_des.compute(roi_in_mat, feature_vec);

    for (int j=0; j<6; j++) {
        cv::Mat ROI_l = GetUsedChannel(roi_in_mat, j);
        cv::Moments moment = cv::moments(ROI_l, false);


        // lbp related
        // cv::Mat lbp_mat;
        // cv::resize(t_image_l, t_image_l, cv::Size(30, 30));
        // calExtendLBPFeature(ROI_l, Size(16, 16), lbp_mat);
        // for (int k=0; k<lbp_mat.cols; k++) {
        //     feature_vec.push_back(lbp_mat.at<float>(0, k));
        // }

        double hu[7];
        cv::HuMoments(moment, hu);
        for (int k=0; k<7; k++) {
            feature_vec.push_back(hu[k]);
        }
        // for (int k=0; k<lbp_vec.cols; k++) {
        //     t_descrip_vec.push_back(lbp_vec.at<uchar>(0, k));
        // }
    }

    cv::Mat t(feature_vec);
    // cout<<t<<endl;
    cv::Mat feature_vec_in_mat = t.t();
    // cout<<feature_vec_in_mat<<endl;
    feature_vec_in_mat.convertTo(feature_vec_in_mat, CV_32FC1);
    return feature_vec_in_mat;
}



void GradVision::LoadParameters(string config_path) {
    std::ifstream in_file(config_path);
    
    if (!in_file) {
        cerr<<"Error:"<<__FILE__
                <<":line"<<__LINE__<<endl
                <<"     Complied on"<<__DATE__
                <<"at"<<__TIME__<<endl;
    }
    int i = 0;
    string line_words;
    cout<<"Loading Parameters"<<endl;
    while (in_file >> line_words) {
        cout<<line_words<<endl;
        std::istringstream ins(line_words);
        switch (i++) {
        case 0:
            ins >> gaus_size_;
            break;
        case 1:
            ins >> channel_idx_;
            break;
        case 2:
            ins >> grad_thre_;
            break;
        case 3:
            ins >> area_thre_;
            break;
        case 4:
            ins >> wh_rate_thre_;
            break;
        case 5:
            ins >> svm_model_name_;
            break;
       }
    }
#ifdef ADJUST_PARAMETER
    svm_classifier_.load(svm_model_name_.c_str());
#else
    svm_classifier_.load(("../source/data/set_sprint_param/"+svm_model_name_).c_str());
#endif
}

void GradVision::StoreParameters() {
    std::ofstream out_file("./7.txt");
    if (!out_file) {
        cerr<<"Error:"<<__FILE__
                <<":line"<<__LINE__<<endl
                <<"     Complied on"<<__DATE__
                <<"at"<<__TIME__<<endl;
    }
    out_file << setw(3) << setfill('0') << gaus_size_                   <<"___gaus_size_"<<endl;
    out_file << setw(3) << setfill('0') << channel_idx_                 <<"___channel_idx_"<<endl;
    out_file << setw(3) << setfill('0') << grad_thre_                   <<"___grad_thre_"<<endl;
    out_file << setw(3) << setfill('0') << area_thre_                   <<"___area_thre_"<<endl;
    out_file << setw(3) << setfill('0') << wh_rate_thre_                <<"___wh_rate_thre_"<<endl;
    out_file << svm_model_name_;
    out_file.close();
}

void GradVision::set_all_parameters(AllParameters_grad ap) {
    gaus_size_      = ap.gaus_size;
    channel_idx_    = ap.channel_idx;
    grad_thre_      = ap.grad_thre;
    area_thre_      = ap.area_thre;
    wh_rate_thre_   = ap.wh_rate_thre;
}
     
void GradVision::WriteImg(cv::Mat src, string folder_name, int num) {
    stringstream t_ss;
    string path = "../source/data/con_img/";
    if (start_file_num_ <= max_file_num_) {
        path += folder_name;
        path += "/";

        t_ss << num;
        path += t_ss.str();
        t_ss.str("");
        t_ss.clear();
        // path += std::to_string(num); 

        path += ".jpg";

        cv::imwrite(path,src);
    }
}

cv::Mat GradVision::GetUsedChannel(cv::Mat& src_img, int flag) {
    cv::Mat t;
    cv::Mat t_cs[3];
    switch (flag) {
    case 0:
    case 1:
    case 2:
        cv::cvtColor(src_img, t, CV_BGR2HSV_FULL);
        cv::split(t, t_cs);
        return t_cs[flag];
    case 3:
    case 4:
    case 5:
        cv::cvtColor(src_img, t, CV_BGR2Lab);
        cv::split(t, t_cs);
        return t_cs[flag - 3];
    }
}
