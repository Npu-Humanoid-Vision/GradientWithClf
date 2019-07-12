#ifndef PARAMS_H
#define PARAMS_H

#include <time.h>
#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#ifdef __linux__
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#endif

#ifdef __WIN32
#include <io.h>
#include <windows.h>
#include <opencv2/nonfree/nonfree.hpp>

#endif

enum { H,S,V,L,A,B };

#define POS_LABLE 1
#define NEG_LABLE 0

#define IMG_COL 128 
#define IMG_ROW 128 

#define MODEL_PATH "./Model/"

#define TESTSET_PATH "./Data/Test/"
#define TRAINSET_PATH "./Data/Train/"

#define RAW_DATA_PATH "./Data/Raw/"

#endif