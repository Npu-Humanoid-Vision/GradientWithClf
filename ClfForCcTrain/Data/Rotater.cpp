#include <Params.h>
#include <Util.h>

#define POS_COUNTER_INIT_NUM 0
#define NEG_COUNTER_INIT_NUM 0
#define SAVE_PATH "./Rotated/"

cv::Mat Rotate(cv::Mat& input_image, double angle);
string GetPath(string save_path, int lable);

int main(int argc, char const *argv[]) {

    string raw_data_path = "./Raw/";

    double rotate_angle[3] = {0, 120, 240};

    std::vector<string> pos_names;
    std::vector<string> neg_names;

    GetImgNames(raw_data_path+"Pos/", pos_names);
    GetImgNames(raw_data_path+"Neg/", neg_names);

    cv::namedWindow("233", CV_WINDOW_NORMAL);
    for (auto i = pos_names.begin(); i != pos_names.end(); i++) {
        cv::Mat t_image = cv::imread(raw_data_path+"Pos/"+*i);
        if (t_image.empty()) {
            continue;
        }

        for (auto j = 0; j < 3; j++) {
            cv::Mat dst = Rotate(t_image, rotate_angle[j]);
            cv::imshow("233", dst);
            cv::waitKey(1);
            cv::imwrite(GetPath(SAVE_PATH, POS_LABLE), dst);
        }
    }
    for (auto i = neg_names.begin(); i != neg_names.end(); i++) {
        // cout<<raw_data_path+"Neg/"+*i<<endl;
        cv::Mat t_image = cv::imread(raw_data_path+"Neg/"+*i);
        if (t_image.empty()) {
            continue;
        }
        for (auto j = 0; j < 3; j++) {
            cv::Mat dst = Rotate(t_image, rotate_angle[j]);
            cv::imshow("233", dst);
            cv::waitKey(1);
            cv::imwrite(GetPath(SAVE_PATH, NEG_LABLE), dst);
        }
    }
    return 0;
}


cv::Mat Rotate(cv::Mat& src, double angle) {
	cv::Mat dst;
	cv::Size dst_sz(src.cols, src.rows);    
	cv::Point2f center(src.cols / 2., src.rows / 2.);
 
	cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);
	cv::warpAffine(src, dst, rot_mat, dst_sz, INTER_LINEAR, BORDER_REPLICATE);
    cv::resize(dst, dst, cv::Size(IMG_COL, IMG_ROW));
    return dst;
}



int pos_counter = POS_COUNTER_INIT_NUM;
int neg_counter = NEG_COUNTER_INIT_NUM;

string GetPath(string save_path, int lable) {
    stringstream t_ss;
    string t_s;

    if (lable == POS_LABLE) {
        save_path += "Pos/";
        t_ss << pos_counter++;
        t_ss >> t_s;
        t_s = save_path + t_s;
        t_s += ".jpg";
        cout<<t_s<<endl;
    }
    else {
        save_path += "Neg/";
        t_ss << neg_counter++;
        t_ss >> t_s;
        t_s = save_path + t_s;
        t_s += ".jpg";
        cout<<t_s<<endl;   
    }
    return t_s;
}