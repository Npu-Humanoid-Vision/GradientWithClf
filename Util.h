#ifndef UTIL_H
#define UTIL_H

#include <opencv2/opencv.hpp>
using namespace cv;


int calAverageGary(const Mat &inImg, int &maxGaryDiff, int &averageGrad_xy);
void ComputeLBPImage_Uniform(const Mat &srcImage, Mat &LBPImage);
void ComputeLBPFeatureVector_Uniform(const Mat &srcImage, Size cellSize, Mat &featureVector);
void calExtendLBPFeature(const Mat &srcImage, Size cellSize, Mat &extendLBPFeature);

 
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