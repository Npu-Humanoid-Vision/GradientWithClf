
#include "GradVision.h"

cv::VideoCapture cp(0);
cv::Mat frame;
GradVision test_vision("./7.txt");
GradVisionResult gabage;
AllParameters_grad all_p;    


// #define RUN_ON_DARWIN 

int main(int argc, char const *argv[]) {
    if (!cp.isOpened()) {
        cerr<<"open camera fail"<<endl;
        return -1;
    }
    int wh_rate_int = 37;// need to /10

    all_p.gaus_size     = test_vision.gaus_size_;
    all_p.channel_idx   = test_vision.channel_idx_;
    all_p.grad_thre     = test_vision.grad_thre_;
    all_p.area_thre     = test_vision.area_thre_;
    all_p.wh_rate_thre  = test_vision.wh_rate_thre_;


    cv::namedWindow("set_params", CV_WINDOW_NORMAL);
    
    cv::createTrackbar("gaus_size", "set_params", &all_p.gaus_size, 255);
    cv::createTrackbar("channel_idx", "set_params", &all_p.channel_idx, 255);
    cv::createTrackbar("grad_thre", "set_params", &all_p.grad_thre, 255);
    cv::createTrackbar("area_thre", "set_params", &all_p.area_thre, 255);
    cv::createTrackbar("wh_rate", "set_params", &wh_rate_int, 120);

    while (1) {
        cp >> frame;
        if (frame.empty()) {
            cerr<<"frame empty"<<endl;
            return -1;
        }



#ifdef RUN_ON_DARWIN
        cv::flip(frame, frame, -1);
        cv::resize(frame, frame, cv::Size(320, 240));
#endif
        all_p.wh_rate_thre = wh_rate_int/10.0;
        test_vision.set_all_parameters(all_p);
        
        test_vision.imageProcess(frame, &gabage);
        

        char key = cv::waitKey(1);
        if (key == 's') {
            test_vision.StoreParameters();
            break;
        }
        else if (key == 'q') {
            break;
        }
    }
    return 0;
}
