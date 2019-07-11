#ifndef UTIL_H
#define UTIL_H

#include <Params.h>

struct EvaluationValues {
    int TP, FP;
    int FN, TN;
    double precision_rate;
    double recall_rate;
    double f1_score;
};



cv::Mat GetUsedChannel(cv::Mat& src_img, int flag);
void GetImgNames(string root_path, std::vector<std::string>& names);
void GetXsSampleData(const string folder_path, int lable, 
            cv::Mat& train_data, cv::Mat& train_data_lables);
string GetNowTime();

// get test_data, return TP, FP, FN, TN num
/*
+————————————————————————————————————————————————————+
|actual               | positive   | actual negative |
|predicted positive   | TP	       | FP              |
|predicted negative   | FN	       | TN              |
+————————————————————————————————————————————————————+
*/
template<class CLF_TYPE>
void GetXX(cv::Mat& test_data, CLF_TYPE& tester, int lable, int& true_num, int& false_num);
void GetScores(string test_data_path, string model_path, EvaluationValues& scores);

string Train_csvc(double C);
void Evaluation(string model_path);

int calAverageGary(const Mat &inImg, int &maxGaryDiff, int &averageGrad_xy);
void ComputeLBPImage_Uniform(const Mat &srcImage, Mat &LBPImage);
void ComputeLBPFeatureVector_Uniform(const Mat &srcImage, Size cellSize, Mat &featureVector);
void calExtendLBPFeature(const Mat &srcImage, Size cellSize, Mat &extendLBPFeature);

// 获得某文件夹下所有图片的名字
void GetImgNames(string root_path, std::vector<std::string>& names) {
#ifdef __linux__
    struct dirent* filename;
    DIR* dir;
    dir = opendir(root_path.c_str());  
    if(NULL == dir) {  
        return;  
    }  

    int iName=0;
    while((filename = readdir(dir)) != NULL) {  
        if( strcmp( filename->d_name , "." ) == 0 ||
            strcmp( filename->d_name , "..") == 0)
            continue;

        string t_s(filename->d_name);
        names.push_back(t_s);
    }
#endif

#ifdef __WIN32
    intptr_t hFile = 0;
    struct _finddata_t fileinfo;
    string p;

    hFile = _findfirst(p.assign(root_path).append("/*").c_str(), &fileinfo);

    if (hFile != -1) {
        do {
            if (strcmp(fileinfo.name, ".") == 0 || strcmp(fileinfo.name, "..") == 0) {
                continue;
            }
            names.push_back(fileinfo.name); 
        } while (_findnext(hFile, &fileinfo) == 0);
    }
#endif
}

cv::Mat GetUsedChannel(cv::Mat& src_img, int flag) {
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


void GetXsSampleData(const string folder_path, int lable, 
            cv::Mat& train_data, cv::Mat& train_data_lables) {

    // get the image names
    std::vector<std::string> image_names;
    GetImgNames(folder_path, image_names);

    // define hog descriptor 
    cv::HOGDescriptor hog_des(Size(128, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);

    // read images and compute
    for (auto i = image_names.begin(); i != image_names.end(); i++) {
        string t_path = folder_path + (*i);
        cv::Mat t_image = cv::imread(t_path);
        std::vector<float> t_descrip_vec;

        // hog related
        hog_des.compute(t_image, t_descrip_vec);

        // moment related
        for (int j=0; j<6; j++) {
            cv::Mat t_image_l = GetUsedChannel(t_image, j);
            cv::Moments moment = cv::moments(t_image_l, false);

            // lbp related
            // cv::Mat lbp_mat;
            // cv::resize(t_image_l, t_image_l, cv::Size(30, 30));
            // calExtendLBPFeature(t_image_l, Size(16, 16), lbp_mat);
            // for (int k=0; k<lbp_mat.cols; k++) {
            //     t_descrip_vec.push_back(lbp_mat.at<float>(0, k));
            // }
            double hu[7];
            cv::HuMoments(moment, hu);
            for (int k=0; k<7; k++) {
                t_descrip_vec.push_back(hu[k]);
            }
            
        }

        // copy t_descrip_vec to train_data
        cv::Mat t_mat = cv::Mat(1, t_descrip_vec.size(), CV_32FC1);
        for (auto j = 0; j < t_descrip_vec.size(); j++) {
            t_mat.at<float>(0, j) = t_descrip_vec[j];
        }
        train_data.push_back(t_mat);
        train_data_lables.push_back(lable);
    }
}

string GetNowTime() {
	time_t t;
	time(&t);
	char tmp[64];
	strftime(tmp, sizeof(tmp), "%m%d_%H%M%S",localtime(&t));
	return tmp;
}

template<class CLF_TYPE>
void GetXX(cv::Mat& test_data, CLF_TYPE& tester, int lable, int& true_num, int& false_num) {
    true_num  = 0;
    false_num = 0;

    // load image for show
    string folder_path = TESTSET_PATH;
    if (lable == POS_LABLE) {
        folder_path += "Pos/";
    }
    else {
        folder_path += "Neg/";
    }
    std::vector<std::string> image_names;
    GetImgNames(folder_path, image_names);


    int test_sample_num = test_data.rows;
    for (auto i = 0; i < test_sample_num; i++) {
        cv::Mat test_vec = test_data.row(i);

#if CV_MAJOR_VERSION < 3

    #ifdef REGRESSION
        double scores = tester.predict(test_vec);
        int t_predict_lable;
        if (scores > 0.5) {
            t_predict_lable = POS_LABLE;
        }
        else {
            t_predict_lable = NEG_LABLE;
        }
    #else
        int t_predict_lable = (int)tester.predict(test_vec);
    #endif
#else 
        cv::Mat lable_mat;
        tester->predict(test_vec, lable_mat);
        int t_predict_lable = lable_mat.at<float>(0, 0);
#endif

        if (t_predict_lable == lable) {
            true_num++;
        }
        else {
            false_num++;
            cout<<"wrong classified: "<<folder_path<<image_names[i]<<endl;
            cv::Mat t = cv::imread(folder_path+image_names[i]);
            cv::imshow("wrong classified", t);
            cv::waitKey(0);
        }
    }

    return ;
}

// 返回各种指标
void GetScores(string test_data_path, string model_path, EvaluationValues& scores) {
    // get test data
    cv::Mat test_data_pos;
    cv::Mat test_data_neg;
    cv::Mat test_data_lables;
    GetXsSampleData(test_data_path+"/Pos/", POS_LABLE, test_data_pos, test_data_lables);
    GetXsSampleData(test_data_path+"/Neg/", NEG_LABLE, test_data_neg, test_data_lables);
    cout<<"test data size: "<<test_data_lables.size()<<endl;

#if CV_MAJOR_VERSION < 3
    // get classifier 
    CvSVM tester;
    tester.load(model_path.c_str());
#else 
    cv::Ptr<cv::ml::SVM> tester = cv::ml::SVM::load(model_path.c_str());
#endif

    // get XX
    GetXX(test_data_pos, tester, POS_LABLE, scores.TP, scores.FN);
    GetXX(test_data_neg, tester, NEG_LABLE, scores.TN, scores.FP);
    cout<<"TP: "<<scores.TP<<"\t FP: "<<scores.FP<<endl
        <<"FN: "<<scores.FN<<"\t TN: "<<scores.TN<<endl;

    scores.precision_rate = 1.0*scores.TP/(scores.TP+scores.FP);
    scores.recall_rate    = 1.0*scores.TP/(scores.TP+scores.FN); 
    scores.f1_score       = 2.0*scores.precision_rate*scores.recall_rate/(scores.precision_rate+scores.recall_rate);

    return ;
}



string Train_csvc(double C) {
	string pos_root_path = string(TRAINSET_PATH) + string("Pos/");
    string neg_root_path = string(TRAINSET_PATH) + string("Neg/");
	
    cout<<pos_root_path<<endl<<neg_root_path<<endl;
    cv::Mat train_data;
    cv::Mat train_data_lables;
    GetXsSampleData(pos_root_path, POS_LABLE, train_data, train_data_lables);
    GetXsSampleData(neg_root_path, NEG_LABLE, train_data, train_data_lables);
    cout<<train_data.size()<<' '<<train_data_lables.size()<<endl;

#ifdef __WIN32 // mingw 只配了 opencv2
    // 参数设置
    CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 50000, FLT_EPSILON);
    CvSVMParams train_params(CvSVM::C_SVC, CvSVM::LINEAR, 0.1, 0.1, 0.1, C, 0.5, 0.5, 0, criteria);

    CvSVM trainer;

    double begin = (double)getTickCount();
	if (C > 0)  {
		CvSVMParams train_params(CvSVM::C_SVC, CvSVM::LINEAR, 0.1, 0.1, 0.1, C, 0.5, 0.5, 0, criteria);
		trainer.train(train_data, train_data_lables, cv::Mat(), cv::Mat(), train_params);
	}
	else {
		CvSVMParams train_params(CvSVM::C_SVC, CvSVM::LINEAR, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0, criteria);
		trainer.train_auto(train_data, train_data_lables, cv::Mat(), cv::Mat(), train_params);
	}

    cout<<"train take time: "<<((double)getTickCount() - begin)/getTickFrequency()<<endl;
    cout<<"C: "<<trainer.get_params().C<<endl;
	string model_path = MODEL_PATH;
	model_path = model_path + "c_svc_" + GetNowTime() + ".xml";
    trainer.save(model_path.c_str());
    cout<<"Save model to: "<<model_path<<endl;
	return model_path;
#endif

}

void Evaluation(string model_path) {
	string test_data_path = TESTSET_PATH;
    EvaluationValues scores;
    GetScores(test_data_path, model_path, scores);
    cout<<"precision rate: \t"<<scores.precision_rate<<endl
        <<"recall rate: \t\t"<<scores.recall_rate<<endl
        <<"f1 scores: \t\t"<<scores.f1_score<<endl;

}

 
//计算输入图片的最大灰度差、平均灰度、平均梯度
int calAverageGary(const Mat &inImg, int &maxGaryDiff, int &averageGrad_xy)
{
	float averageGary;
	int garySum = 0;
	int i, j;
 
	//求平均灰度值
	for (i=0; i<inImg.cols; i++)
	{
		for (j=0; j<inImg.rows; j++)
		{
			garySum += inImg.at<uchar>(j, i);
		}
	}
	averageGary = (int)(garySum*1.0f/(inImg.rows*inImg.cols));
 
	//求滑窗内的最大灰度差值
	double minGary, maxGary; 
	minMaxLoc(inImg, &minGary, &maxGary, NULL, NULL);
	maxGaryDiff = (int)(maxGary-minGary);
 
	//求滑窗内的平均梯度值
	Mat grad_x, grad_y, abs_grad_x, abs_grad_y, grad_xy; 
	Sobel( inImg, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT );  	//求X方向梯度 
	convertScaleAbs( grad_x, abs_grad_x );  
	Sobel( inImg, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT );  	//求Y方向梯度  
	convertScaleAbs( grad_y, abs_grad_y );  
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad_xy);  	       //合并梯度(近似)  
	//cout<<"gary_xy"<<grad_xy<<endl;
 
	int grad_xy_sum = 0;
	for (i=0; i<inImg.cols; i++)
	{
		for (j=0; j<inImg.rows; j++)
		{
			grad_xy_sum += grad_xy.at<uchar>(j, i);
		}
	}
	averageGrad_xy = (int)(grad_xy_sum*1.0f/(inImg.rows*inImg.cols));
	return averageGary;
}
 
 
// 计算等价模式LBP特征图
void ComputeLBPImage_Uniform(const Mat &srcImage, Mat &LBPImage)
{
	// 参数检查，内存分配
	CV_Assert(srcImage.depth() == CV_8U&&srcImage.channels() == 1);
	LBPImage.create(srcImage.size(), srcImage.type());
 
	// 计算LBP图
	Mat extendedImage;
	copyMakeBorder(srcImage, extendedImage, 1, 1, 1, 1, BORDER_DEFAULT);
 
	// LUT(256种每一种模式对应的等价模式)
	static const int table[256] = { 1, 2, 3, 4, 5, 0, 6, 7, 8, 0, 0, 0, 9, 0, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 14, 0, 15, 16, 17, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 20, 0, 21, 22, 23, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25,
		0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 27, 0, 28, 29, 30, 31, 0, 32, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 34, 0, 0, 0, 0
		, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 36, 37, 38, 0, 39, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42
		, 43, 44, 0, 45, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 47, 48, 49, 0, 50, 0, 0, 0, 51, 52, 53, 0, 54, 55, 56, 57, 58 };
 
	// 计算LBP
	int heightOfExtendedImage = extendedImage.rows;
	int widthOfExtendedImage = extendedImage.cols;
	int widthOfLBP=LBPImage.cols;
	uchar *rowOfExtendedImage = extendedImage.data+widthOfExtendedImage+1;
	uchar *rowOfLBPImage = LBPImage.data;
 
	int pixelDiff = 5;
 
	for (int y = 1; y <= heightOfExtendedImage - 2; ++y,rowOfExtendedImage += widthOfExtendedImage, rowOfLBPImage += widthOfLBP)
	{
		// 列
		uchar *colOfExtendedImage = rowOfExtendedImage;
		uchar *colOfLBPImage = rowOfLBPImage;
		for (int x = 1; x <= widthOfExtendedImage - 2; ++x, ++colOfExtendedImage, ++colOfLBPImage)
		{
			// 计算LBP值
			int LBPValue = 0;
			if (colOfExtendedImage[0 - widthOfExtendedImage - 1] >= colOfExtendedImage[0]+pixelDiff)
				LBPValue += 128;
			if (colOfExtendedImage[0 - widthOfExtendedImage] >= colOfExtendedImage[0]+pixelDiff)
				LBPValue += 64;
			if (colOfExtendedImage[0 - widthOfExtendedImage + 1] >= colOfExtendedImage[0]+pixelDiff)
				LBPValue += 32;
			if (colOfExtendedImage[0 + 1] >= colOfExtendedImage[0]+pixelDiff)
				LBPValue += 16;
			if (colOfExtendedImage[0 + widthOfExtendedImage + 1] >= colOfExtendedImage[0]+pixelDiff)
				LBPValue += 8;
			if (colOfExtendedImage[0 + widthOfExtendedImage] >= colOfExtendedImage[0]+pixelDiff)
				LBPValue += 4;
			if (colOfExtendedImage[0 + widthOfExtendedImage - 1] >= colOfExtendedImage[0]+pixelDiff)
				LBPValue += 2;
			if (colOfExtendedImage[0 - 1] >= colOfExtendedImage[0]+pixelDiff)
				LBPValue += 1;
 
			colOfLBPImage[0] = table[LBPValue];
		}
	}
}
 
//计算归一化的LBP特征矩阵
void ComputeLBPFeatureVector_Uniform(const Mat &srcImage, Size cellSize, Mat &featureVector)
{
	// 参数检查，内存分配
	CV_Assert(srcImage.depth() == CV_8U&&srcImage.channels() == 1);
 
	Mat LBPImage;
	ComputeLBPImage_Uniform(srcImage, LBPImage);
 
	//cout<<"LBPImage_uniform："<<endl<<LBPImage<<endl<<endl;
 
	// 计算cell个数
	int widthOfCell = cellSize.width;
	int heightOfCell = cellSize.height;
	int numberOfCell_X = srcImage.cols / widthOfCell;// X方向cell的个数
	int numberOfCell_Y = srcImage.rows / heightOfCell;
 
	// 特征向量的个数
	int numberOfDimension = 58 * numberOfCell_X*numberOfCell_Y;
	featureVector.create(1, numberOfDimension, CV_32FC1);
	featureVector.setTo(Scalar(0));
 
	// 计算LBP特征向量
	int stepOfCell=srcImage.cols;
	int index = -58;// cell的特征向量在最终特征向量中的起始位置
	float *dataOfFeatureVector=(float *)featureVector.data;
	for (int y = 0; y <= numberOfCell_Y - 1; ++y)
	{
		for (int x = 0; x <= numberOfCell_X - 1; ++x)
		{
			index+=58;
 
			// 计算每个cell的LBP直方图
			Mat cell = LBPImage(Rect(x * widthOfCell, y * heightOfCell, widthOfCell, heightOfCell));
			uchar *rowOfCell=cell.data;
			int sum = 0; // 每个cell的等价模式总数
			for(int y_Cell=0;y_Cell<=cell.rows-1;++y_Cell,rowOfCell+=stepOfCell)
			{
				uchar *colOfCell=rowOfCell;
				for(int x_Cell=0;x_Cell<=cell.cols-1;++x_Cell,++colOfCell)
				{
					if(colOfCell[0]!=0)
					{
						// 在直方图中转化为0~57，所以是colOfCell[0] - 1
						++dataOfFeatureVector[index + colOfCell[0]-1];
						++sum;
					}
				}
			}
 
			for (int i = 0; i <= 57; ++i)
				dataOfFeatureVector[index + i] /= sum;
		}
	}
}
 
//计算扩展LBP特征矩阵，在原LBP特征上增加了3维
void calExtendLBPFeature(const Mat &srcImage, Size cellSize, Mat &extendLBPFeature)
{
	// 参数检查，内存分配
	CV_Assert(srcImage.depth() == CV_8U&&srcImage.channels() == 1);
 
	Mat LBPImage;
	int i,j, height, width;
	height = srcImage.rows;
	width = srcImage.cols;
 
	ComputeLBPFeatureVector_Uniform(srcImage, cellSize, LBPImage);   //求归一化后的LBP特征
	//cout<<"LBPImage"<<LBPImage<<endl;   //取值范围[0,58]
 
	//把LBPImage折算到[0,255]之间
	Mat LBPImage_255(1, LBPImage.cols, CV_8UC1, Scalar(0));
	for (i=0; i<LBPImage.cols; i++)
	{
		LBPImage_255.at<uchar>(0,i) = (uchar)(LBPImage.at<float>(0,i) * 255.0f);
	}
	//cout<<"LBPImage_255"<<endl<<LBPImage_255<<endl;
 
	int maxGaryDiff, averageGrad_xy;
	int averageGary = calAverageGary(srcImage, maxGaryDiff, averageGrad_xy);
	//cout<<"averageGary="<<averageGary<<",   maxGrayDiff="<<maxGaryDiff<<endl<<endl;
 
	int descriptorDim;
	descriptorDim = LBPImage.cols + 3;
	Mat extendLBPFeature_255 = Mat::zeros(1, descriptorDim, CV_8UC1); 
 
	for (i=0; i<LBPImage.cols; i++)
	{
		extendLBPFeature_255.at<uchar>(0,i) = LBPImage_255.at<uchar>(0,i);
	}
	extendLBPFeature_255.at<uchar>(0,LBPImage.cols) = averageGary;       //增加维度，存放平均像素
	extendLBPFeature_255.at<uchar>(0,LBPImage.cols+1) = maxGaryDiff;     //增加维度，存放最大灰度差
	extendLBPFeature_255.at<uchar>(0,LBPImage.cols+2) = averageGrad_xy;  //增加维度，存放平均梯度
 
	//把扩展LBP特征矩阵归一化
	extendLBPFeature = Mat(1, descriptorDim, CV_32FC1, Scalar(0)); 
	for(i=0; i<descriptorDim; i++)
	{
		extendLBPFeature.at<float>(0,i) = extendLBPFeature_255.at<uchar>(0,i)*1.0f/255;
	}
	//cout<<"extendLBPFeature： "<<endl<<extendLBPFeature<<endl;
}



#endif