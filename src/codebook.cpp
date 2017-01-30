#include "../header/codebook.hpp"

Codebook::Codebook(int weightSize) {
	mWeightSize = weightSize;
	mWeightSizeOf = weightSize*sizeof(double);
}

void Codebook::setTrainingSet(double *trainingSet, int length, int weightSize) {
	mWeightSize = weightSize;
	mWeightSizeOf = weightSize*sizeof(double);
	// mTrainingSet = new vector<double *>();

	for (int i = 0; i < length; ++i) {
		double *input = (double *) malloc(mWeightSizeOf);
		memcpy(input, &trainingSet[i * mWeightSize], mWeightSizeOf);
		mTrainingSet.push_back(input);
	}
	mDistances = new vector<int>(length, -1);
	mBMU = new vector<int>(length, -1);
	mMappedBMU = new vector<int>(length, -1); 
	// mTrainingSet = (double *) malloc(length);
	// memset(mTrainingSet, trainingSet, length);
}

void Codebook::setTrainingSet(double *trainingSet, int length) {
	for (int i = 0; i < length; ++i) {
		double *input = (double *) malloc(mWeightSizeOf);
		memcpy(input, &trainingSet[i * mWeightSize], mWeightSizeOf);
		mTrainingSet.push_back(input);
	}
	mDistances = new vector<int>(length, -1);
	mBMU = new vector<int>(length, -1);
	mMappedBMU = new vector<int>(length, -1); 
	// mTrainingSet = (double *) malloc(length);
	// memset(mTrainingSet, trainingSet, length);
}

void Codebook::getTrainingSet(double *trainingSet) {

}

void Codebook::setDistance(double *distance, int position) {
	mDistances.insert(position, distance);
}

void Codebook::getDistance(double *distance, int position) {

}

void Codebook::setBMU(int bmu, int position) {
	mBMU.insert(position, bmu);
}

void Codebook::getBMU(int &bmu, int position) {

}

void Codebook::setMappedBMU(int bmu, int position) {
	mMappedBMU.insert(position, bmu);
}

void Codebook::getMappedBMU(int *bmu, int position) {

}