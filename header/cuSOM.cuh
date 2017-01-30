#ifndef CUSOM_CUH
#define CUSOM_CUH

#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "global.cuh"

using namespace std;
using namespace cv;

enum SOM_DATA_TYPE { UNIFORM, 
	UNIFORM_DOUBLE,
	UINT8,
	UINT16,
	UINT32,
	UINT64 
};

class cuSOM {

public:
	cuSOM(int, int, int, int = 100, double = 0.1, int type = UNIFORM_DOUBLE, bool = true, int = 0);

	void setEpoch(int);
	void setRadius(double);

	void train(bool = false, int = 1);
	int getBMU();
	void evaluate(double*, int&, int&, int*);
	void printMap();
	void printTrainingSet();
	int getMapSize();
	int getMapNeuronSize();
	void printMapDifferences();
	void setTrainingSet(double*, int);
	void showColorMap(int = 1, int = 0, string = "");
	void printMatlabMap();
	void inputQuantizationError();
	void inputTopographicError();
	void quantizationError(double *);

	~cuSOM();

private:
	int mRows;
	int mCols;
	int mWeightSize;

	int mThreadsBlockRows;
	int mThreadsBlockCols;
	int mGridRows;

	int mEpoch;
	double mRadius;
	int mNeuronsSize;
	double *mMap_host;
	double *mMap_device;
	int mMapSizeOf;
	double mLearningRate;
	double *mTrainingSet;
	int mTrainingSetLength;
	cublasHandle_t mCublasHandler;
	int mType;

	double *mStartingMap;

	void initMap(int type = UINT8, int = 0);
	void initCudaLibraries();
	double neighborhoodRadiusDecay(double, int);
	void setNeuronsSize();
};

#endif