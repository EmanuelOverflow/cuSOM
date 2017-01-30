#ifndef SERIALSOM_HPP
#define SERIALSOM_HPP

#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>

using namespace std;
using namespace chrono;
// using namespace cv;

enum SOM_DATA_TYPE { UNIFORM, 
	UNIFORM_DOUBLE,
	UINT8,
	UINT16,
	UINT32,
	UINT64 
};

class serialSOM {

public:
	serialSOM(int, int, int, int = 100, double = 0.1, int type = UNIFORM_DOUBLE, bool = true, int = 0);
	// cuSOM(int, int, int, int);
	// cuSOM(int, int, int, int, int);

	void setEpoch(int);
	void setRadius(double);

	void train(bool = false, int = 1);
	int getBMU();
	void evaluate(double*, int&, int&, int*); // pass input to evaluate?
	void printMap();
	void printTrainingSet();
	int getMapSize();
	int getMapNeuronSize();
	void printMapDifferences();
	void setTrainingSet(double*, int);
	// void showColorMap(int = 100, string = "");
	void printMatlabMap();

	~serialSOM();

private:
	int mRows;
	int mCols;
	int mWeightSize;

	int mEpoch;
	double mRadius;
	int mNeuronsSize;
	double *mMap_host;
	int mMapSizeOf;
	double mLearningRate;
	double *mTrainingSet;
	int mTrainingSetLength;

	int mType;

	double *mStartingMap;

	void initMap(int type = UINT8, int = 0);
	double neighborhoodRadiusDecay(double, int);
	void setNeuronsSize();
	void generateUniformRandomArray(double *, int);
	double learningDecay(double, int, int);
	double neighborhoodFunction(double, int, double);
};

#endif