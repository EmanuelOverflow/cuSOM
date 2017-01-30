#ifndef MAIN_CUH
#define MAIN_CUH

#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>
#include <cmath>
#include "cuSOM.cuh"
#include "global.cuh"
#include "cmdline.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

void readCSVDataset(int, int, double *);
void showColorInput(double *);

#endif