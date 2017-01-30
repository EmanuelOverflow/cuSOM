#include "../header/serialSOM.hpp"

// MAGARI PROVAMI PRIMA DI FARE I TEST
serialSOM::serialSOM(int rows, int cols, int size, int epoch, double learningRate, int type, bool randomInit, int device) : mRows(rows), mCols(cols), mWeightSize(size), mEpoch(epoch), mLearningRate(learningRate) {
	setNeuronsSize();
	mMapSizeOf = mNeuronsSize*sizeof(double);
	mMap_host = (double *) malloc(mMapSizeOf);
	mRadius = min(mRows, mCols); // FIXME
	if (randomInit) {
		initMap(type, device);
	}
	mStartingMap = (double *) malloc(mMapSizeOf);
	memcpy(mStartingMap, mMap_host, mMapSizeOf);	
}

void serialSOM::setNeuronsSize() {
	mNeuronsSize = mCols*mRows*mWeightSize;
}

void serialSOM::setEpoch(int epoch) {
	mEpoch = epoch;
}

// AGGIUSTARE SIZE BLOCCHI E THREAD
void serialSOM::train(bool show, int cellSize) {
	int weightSizeOf = mWeightSize*sizeof(double);
	int distanceSizeOf = getMapSize()*sizeof(double);

	double *input_host = (double *) malloc(weightSizeOf);
	double *distance_host = (double *) malloc(distanceSizeOf);
	
	cout << "Mapsize: " << getMapSize() << endl;
	cout << "Full map/neurons size: " << mNeuronsSize << endl; 
	cout << "Number of epochs: " << mEpoch << endl;
	cout << "Number of training set elements: " <<  mTrainingSetLength << endl;
	cout << "Learning rate: " << mLearningRate << endl;

	// double findbmu_duration = 0.0;
	// double bmudist_duration = 0.0;
	// double adap_duration = 0.0;

	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	for (int e = 0; e < mEpoch; ++e) {
		double currentRadius = neighborhoodRadiusDecay(mRadius, e);
		double radiusSquare = currentRadius*currentRadius;
		double decay = learningDecay(mLearningRate, e, mEpoch);

		// cout << "Epoch: " << e+1 << "/" << mEpoch << endl;
		// cout << "Current radius: " << currentRadius << " Square: " << radiusSquare << endl;

		for (int i = 0; i < mTrainingSetLength; ++i) {
			memset(distance_host, 0, distanceSizeOf);
			memset(input_host, 0, weightSizeOf);
			memcpy(input_host, &mTrainingSet[i * mWeightSize], weightSizeOf);
			
			// FIND BMU
			// high_resolution_clock::time_point findbmu_start = high_resolution_clock::now();
			
			for (int r = 0; r < mRows; ++r) {
				for (int c = 0; c < mCols; ++c) {
					for (int w = 0; w < mWeightSize; ++w) {
						distance_host[r * mCols + c] += mMap_host[r * mCols * mWeightSize + (c * mWeightSize) + w] - input_host[w];
						distance_host[r * mCols + c] *= distance_host[r * mCols + c];
					}
				}
			}

			// high_resolution_clock::time_point findbmu_end = high_resolution_clock::now();

			// findbmu_duration += duration_cast<microseconds>( findbmu_end - findbmu_start ).count();

			double *bmuPosD = min_element(distance_host, distance_host+getMapSize());
			int bmuPos = (int) *bmuPosD;
			// cout << "\nMin " << bmuPos << endl;

			int bmuColIdx = bmuPos % mCols;
			int bmuRowIdx = (bmuPos - bmuColIdx) / mCols;  

			// ADAPTATION
			// high_resolution_clock::time_point bmudist_start = high_resolution_clock::now();

			memset(distance_host, 0, distanceSizeOf);
			for (int r = 0; r < mRows; ++r) {
				for (int c = 0; c < mCols; ++c) {
					distance_host[r * mCols + c] = pow(r - bmuRowIdx, 2) + pow(c - bmuColIdx, 2);
				}
			}

			// high_resolution_clock::time_point bmudist_end = high_resolution_clock::now();
			// bmudist_duration += duration_cast<microseconds>( bmudist_end - bmudist_start ).count();


			// high_resolution_clock::time_point adap_start = high_resolution_clock::now();

			for (int r = 0; r < mRows; ++r) {
				for (int c = 0; c < mCols; ++c) {
					for (int w = 0; w < mWeightSize; ++w) {
						if (distance_host[r * mCols + c] < radiusSquare) {
							double influence = neighborhoodFunction(distance_host[r * mCols + c], radiusSquare, mLearningRate);
							double newWeightEl = input_host[w] - mMap_host[r * mCols * mWeightSize + (c * mWeightSize) + w];
							mMap_host[r * mCols * mWeightSize + (c * mWeightSize) + w] += influence * decay + newWeightEl;
						}
					}
				}
			}

			// high_resolution_clock::time_point adap_end = high_resolution_clock::now();
			// adap_duration += duration_cast<microseconds>( adap_end - adap_start ).count();
		}		
	}

	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>( t2 - t1 ).count();

	// float d = (mTrainingSetLength * mEpoch);

 //    cout << "Find bmu avg duration: " << findbmu_duration/d << "ms" << endl;
 //    cout << "Bmu dists avg duration: " << bmudist_duration/d << "ms" << endl;
    // cout << "Adaptation avg duration: " << adap_duration/d << "ms" << endl;
    
    cout << "Duration: " << duration << "ms" << endl;
    

	free(input_host);
	free(distance_host);
}

int serialSOM::getBMU() {
	return -1;
}

void serialSOM::evaluate(double *input_host, int &bmu_number, int &bmu_map_position, int *bmu_row_col_position) {
	cout << endl << "Input to test" << endl;
	for (int p = 0; p < mWeightSize; ++p) {
		cout << input_host[p] << "\t";
	} cout << endl << endl;

	int weightSizeOf = mWeightSize*sizeof(double);
	int distanceSizeOf = getMapSize()*sizeof(double);

	double *distance_host = (double *) malloc(distanceSizeOf);
 
	// FIND BMU
	for (int r = 0; r < mRows; ++r) {
		for (int c = 0; c < mCols; ++c) {
			for (int w = 0; w < mWeightSize; ++w) {
				distance_host[r * mCols + c] += mMap_host[r * mCols * mWeightSize + (c * mWeightSize) + w] - input_host[w];
				distance_host[r * mCols + c] *= distance_host[r * mCols + c];
			}
		}
	}

	double *bmuPosD = min_element(distance_host, distance_host+getMapSize());
	int bmuPos = (int) *bmuPosD;
	// printf("\nMin %d\n", minDist);

	int bmuColIdx = bmuPos % mCols; // Leave n, do not use z*n;
	int bmuRowIdx = (bmuPos - bmuColIdx) / mCols;  

	bmu_number = bmuPos;
	bmu_row_col_position[0] = bmuColIdx;
	bmu_row_col_position[1] = bmuRowIdx;
	bmu_map_position = bmuRowIdx * mCols * mWeightSize + bmuColIdx * mWeightSize;
}

void serialSOM::printMap() {
	cout << "\nMap\n" << endl;
	//printf("\nMap\n\n");
	for (int i = 0; i < mRows; ++i) {
		for (int j = 0; j < mCols; ++j) {
			cout << "Neuron: " << (i * mCols + j) + 1 << endl; 
			for (int k = 0; k < mWeightSize; ++k) {
				cout << mMap_host[i * mCols * mWeightSize + (j * mWeightSize) + k] << "\t";
			}
			cout << endl;
		}
		cout << endl << endl;
	}
}

void serialSOM::printMapDifferences() {
	cout << "\nMap differences\n" << endl;
	//printf("\nMap\n\n");
	for (int i = 0; i < mRows; ++i) {
		for (int j = 0; j < mCols; ++j) {
			cout << "Neuron: " << (i * mCols + j) + 1 << endl; 
			for (int k = 0; k < mWeightSize; ++k) {
				double diff = mStartingMap[i * mCols * mWeightSize + (j * mWeightSize) + k] - mMap_host[i * mCols * mWeightSize + (j * mWeightSize) + k];
				cout << abs(diff) << "\t";
			}
			cout << endl;
		}
		cout << endl << endl;
	}
}

void serialSOM::printTrainingSet() {
	cout << "\nTrainingSet\n" << endl;

	for (int i = 0; i < mTrainingSetLength; ++i) {
		cout << "Sample " << (i+1) << endl;
		for (int j = 0; j < mWeightSize; ++j) {
			cout << mTrainingSet[i * mWeightSize + j] << "\t";
		}
		cout << endl << endl;
	} 
}

void serialSOM::printMatlabMap() {
	cout << "\nMap\n" << endl;
	cout << "[";
	for (int i = 0; i < mRows; ++i) {
		for (int j = 0; j < mCols; ++j) {
			for (int k = 0; k < mWeightSize; ++k) {
				cout << mMap_host[i * mCols * mWeightSize + (j * mWeightSize) + k] << " ";
			}
			cout << ";";
		}
	} cout << "]";
	cout << endl;
}

int serialSOM::getMapSize() {
	return mRows*mCols;
}

int serialSOM::getMapNeuronSize() {
	return mNeuronsSize;
}

void serialSOM::setTrainingSet(double *trainingSet, int length) {
	mTrainingSet = (double *) malloc(length*mWeightSize*sizeof(double));
	memcpy(mTrainingSet, trainingSet, length*mWeightSize*sizeof(double));
	mTrainingSetLength = length;
}

void serialSOM::initMap(int type, int device) {
	mType = type;
	generateUniformRandomArray(mMap_host, mNeuronsSize);
}

double serialSOM::neighborhoodRadiusDecay(double sigma, int epoch) {
	double lambda = ((double) mEpoch)/log(sigma);
	double e = exp(-((double) epoch / lambda));
	return sigma * e;
}

// void serialSOM::showColorMap(int cellSize, string name) {
// 	if (mWeightSize > 3) {
// 		cout << "More than 3 values" << endl;
// 	} else {
// 		Mat colorMap(mRows*cellSize, mCols*cellSize, CV_32FC3);
// 			for (int i = 0; i < mRows; ++i) {
// 				int startX = i * cellSize;
// 				for (int j = 0; j < mCols; ++j) {
// 					int startY = j * cellSize; 
// 					int bIdx = i * mCols * mWeightSize + (j * mWeightSize);
// 					int gIdx = i * mCols * mWeightSize + (j * mWeightSize) + 1;
// 					int rIdx = i * mCols * mWeightSize + (j * mWeightSize) + 2; 
// 					rectangle(colorMap, Point(startX, startY), Point(startX + cellSize, startY + cellSize), Scalar(mMap_host[bIdx], mMap_host[gIdx], mMap_host[rIdx]), CV_FILLED);
// 				}
// 			}
// 		imshow("SOM Color Map: " + name, colorMap);
// 		waitKey(0);
// 	}
// }

double serialSOM::learningDecay(double learningRate, int currentEpoch, int totalEpoch) {
	return learningRate * exp(- (double) currentEpoch / totalEpoch);
}

double serialSOM::neighborhoodFunction(double distance, int radiusSquare, double learningRate) {
	return exp(-(double) (distance) / (2*radiusSquare));
}

void serialSOM::generateUniformRandomArray(double *input, int size) {
  	for (int i = 0; i < size; ++i) {
  		input[i] = rand() / (RAND_MAX + 1.0);
  	}
}

serialSOM::~serialSOM() {
	free(mMap_host);
	free(mTrainingSet);
}