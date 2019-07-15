#include <Params.h>
#include <Util.h>


string GetPath(string save_path, int lable);

int main(int argc, char const *argv[]) {
    string m_p;
    CvSVM tester;

    DataPretreat();
    m_p = Train_csvc(0.0);
    Evaluation(m_p);

    tester.load(m_p.c_str());
    
    cv::VideoCapture cp;
    if (argc > 1) {
        cout<<argv[0]<<' '<<argv[1]<<endl;
        cp.open((int)(argv[1][0]-'0'));
    }
    else {
        cp.open(0);
    }   
    cv::Mat frame; 
    int roi_rect_x = 100;
    int roi_rect_y = 100;
    int roi_rect_col = 2*IMG_COL;
    int roi_rect_row = 2*IMG_ROW;
    cv::Rect ROI_Rect(roi_rect_x, roi_rect_y, roi_rect_col, roi_rect_row);

    cp >> frame;
    while (frame.empty()) {
        cp >> frame;
    }

    while (1) {
        cp >> frame;
        if (frame.empty()) {
            cerr << __LINE__ <<"frame empty"<<endl;
            return -1;
        }
#ifdef RUN_ON_DARWIN
        cv::flip(frame, frame, -1);
        cv::resize(frame, frame, cv::Size(320, 240));
#endif
        cv::Mat ROI = frame(ROI_Rect).clone();
        cv::resize(ROI, ROI, cv::Size(IMG_COL, IMG_ROW));

        cv::HOGDescriptor hog_des(Size(IMG_COL, IMG_ROW), Size(16,16), Size(8,8), Size(8,8), 9);
        std::vector<float> hog_vec;
        hog_des.compute(ROI, hog_vec);

        for (int j=0; j<6; j++) {
            cv::Mat ROI_l = GetUsedChannel(ROI, j);
            cv::Moments moment = cv::moments(ROI_l, false);

            // cv::Mat lbp_mat;
            // cv::resize(t_image_l, t_image_l, cv::Size(30, 30));
            // calExtendLBPFeature(ROI_l, Size(16, 16), lbp_mat);
            // for (int k=0; k<lbp_mat.cols; k++) {
                // hog_vec.push_back(lbp_mat.at<float>(0, k));
            // }

            double hu[7];
            cv::HuMoments(moment, hu);
            for (int k=0; k<7; k++) {
                hog_vec.push_back(hu[k]);
            }
            // for (int k=0; k<lbp_vec.cols; k++) {
            //     t_descrip_vec.push_back(lbp_vec.at<uchar>(0, k));
            // }
        }
        // cout<<hog_vec.size()<<endl;
        cv::Mat t(hog_vec);
        cv::Mat hog_vec_in_mat = t.t();
        hog_vec_in_mat.convertTo(hog_vec_in_mat, CV_32FC1);

        int lable = (int)tester.predict(hog_vec_in_mat);
        // cout<<tester.predict(hog_vec_in_mat)<<endl;
        if (lable == POS_LABLE) {
            cv::rectangle(frame, ROI_Rect, cv::Scalar(0, 255, 0), 2);
        }
        else {
            cv::rectangle(frame, ROI_Rect, cv::Scalar(0, 0, 255), 2);
        }
        cv::imshow("frame", frame);
        cv::imshow("roi", frame(ROI_Rect));
        char key = cv::waitKey(20);
        if (key == 'q') {
            break;
        }
        else if (key == 'p') {
            cv::imwrite(GetPath(RAW_DATA_PATH, POS_LABLE), ROI);
        }
        else if (key == 'n') {
            cv::imwrite(GetPath(RAW_DATA_PATH, NEG_LABLE), ROI);
        }
        else if (key == 'i') {
            roi_rect_col -= 15;
            roi_rect_row -= 15;
            ROI_Rect = cv::Rect(roi_rect_x, roi_rect_y, roi_rect_col, roi_rect_row);
        }
        else if (key == 'o') {
            roi_rect_col += 15;
            roi_rect_row += 15;
            ROI_Rect = cv::Rect(roi_rect_x, roi_rect_y, roi_rect_col, roi_rect_row);
        }
        else if (key == 'w') {
            roi_rect_y -= 15;
            ROI_Rect = cv::Rect(roi_rect_x, roi_rect_y, roi_rect_col, roi_rect_row);
        }
        else if (key == 's') {
            roi_rect_y += 15;
            ROI_Rect = cv::Rect(roi_rect_x, roi_rect_y, roi_rect_col, roi_rect_row);
        }
        else if (key == 'a') {
            roi_rect_x -= 15;
            ROI_Rect = cv::Rect(roi_rect_x, roi_rect_y, roi_rect_col, roi_rect_row);
        }
        else if (key == 'd') {
            roi_rect_x += 15;
            ROI_Rect = cv::Rect(roi_rect_x, roi_rect_y, roi_rect_col, roi_rect_row);
        }
        else if (key == 'z') {
            roi_rect_col += 15;
            ROI_Rect = cv::Rect(roi_rect_x, roi_rect_y, roi_rect_col, roi_rect_row);
        }
        else if (key == 'x') {
            roi_rect_row += 15;
            ROI_Rect = cv::Rect(roi_rect_x, roi_rect_y, roi_rect_col, roi_rect_row);
        }
        else if (key == 'r') {
            cv::destroyAllWindows();
            DataPretreat();
            m_p = Train_csvc(0.0);
            Evaluation(m_p);

            tester.load(m_p.c_str());
        }
    }

    return 0;
}

string GetPath(string save_path, int lable) {
    stringstream t_ss;
    string t_s;

    if (lable == POS_LABLE) {
        save_path += "Pos/";
        t_ss << GetNowTime();
        t_ss >> t_s;
        t_s = save_path + t_s;
        t_s += ".jpg";
        cout<<t_s<<endl;
    }
    else if(lable == NEG_LABLE) {
        save_path += "Neg/";
        t_ss << GetNowTime();
        t_ss >> t_s;
        t_s = save_path + t_s;
        t_s += ".jpg";
        cout<<t_s<<endl;   
    }
    return t_s;
}


