#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

int g_thre = 10;
int k_size = 1;


int main(int argc, char const *argv[]) {
    cv::VideoCapture cp(0);
    cv::Mat frame;

    cv::namedWindow("grad_thre");
    cv::createTrackbar("thre", "grad_thre", &g_thre, 255);
    cv::createTrackbar("k_size", "grad_thre", &k_size, 12);
    while (1) {
        cp >> frame;
        if (frame.empty()) {
            cerr<<"frame empty"<<endl;
            return -1;
        }
        cv::GaussianBlur(frame, frame, cv::Size(k_size*2+1, k_size*2+1), 0., 0.);

        cv::Mat t_lab;
        cv::Mat t_lab_cs[3];
        cv::cvtColor(frame, t_lab, CV_BGR2Lab);
        cv::split(t_lab, t_lab_cs);

        cv::Mat grad_x;
        cv::Mat grad_y;

        cv::Sobel(t_lab_cs[1], grad_x, CV_16S, 1, 0);
        cv::Sobel(t_lab_cs[1], grad_y, CV_16S, 0, 1);
        cv::convertScaleAbs(grad_x, grad_x);
        cv::convertScaleAbs(grad_y, grad_y);

        cv::Mat grad = 0.5*grad_x+0.5*grad_y;
        
        cv::Mat grad_thre = grad>g_thre;

        std::vector<std::vector<cv::Point> > contours;
        std::vector<std::vector<cv::Point> > contours_poly;
        std::vector<cv::Rect> bound_rect;

        cv::Mat image_for_contours = grad_thre.clone();
        cv::findContours(image_for_contours, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
        contours_poly.resize(contours.size());


        int t = 0;
        for (auto i=0; i<contours.size(); i++) {
            cv::approxPolyDP(contours[i], contours_poly[i], 3, false);
            cv::Rect t_rect = cv::boundingRect(contours_poly[i]);
            if (t_rect.area() < 20)
                continue;
            t++;
            cv::rectangle(frame, t_rect, cv::Scalar(0, 0, 255), 3);
        }
        cout<<"nums of roi: "<<t<<endl;

        cv::imshow("frame", frame);
        cv::imshow("grad", grad);
        cv::imshow("grad_thre", grad_thre);
        char t_key = cv::waitKey(1);
        if (t_key == 'q') 
            break;
    }
    return 0;
}