#include "../header/cuSOM.cuh"

cuSOM::cuSOM(int rows, int cols, int size, int epoch, double learningRate, int type, bool randomInit, int device) : mRows(rows), mCols(cols), mWeightSize(size), mEpoch(epoch), mLearningRate(learningRate) {
	setNeuronsSize();
	mMapSizeOf = mNeuronsSize*sizeof(double);
	mMap_host = (double *) malloc(mMapSizeOf);
	mRadius = min(mRows, mCols);
	initCudaLibraries();
	if (randomInit) {
		initMap(type, device);
	}

	mStartingMap = (double *) malloc(mMapSizeOf);
	memcpy(mStartingMap, mMap_host, mMapSizeOf);

}

void cuSOM::initCudaLibraries() {
	cublasCreate(&mCublasHandler);
}

void cuSOM::setNeuronsSize() {
	mNeuronsSize = mCols*mRows*mWeightSize;
}

void cuSOM::setEpoch(int epoch) {
	mEpoch = epoch;
}

// void cuSOM::hexagonalGrid() {
// 	double xFactor = sqrt(3) / 2;
// }

/*
% Generate hexagonal grid
Rad3Over2 = sqrt(3) / 2;
[X Y] = meshgrid(0:1:41);
n = size(X,1); // number or rows
X = Rad3Over2 * X;
Y = Y + repmat([0 0.5],[n,n/2]);

% Plot the hexagonal mesh, including cell borders
[XV YV] = voronoi(X(:),Y(:)); plot(XV,YV,'b-')
axis equal, axis([10 20 10 20]), zoom on
*/

void cuSOM::train(bool show, int cellSize) {
	cout << "SOM training" << endl;

	int minDist; 
	dim3 threadsPerBlockDistance(mThreadsBlockRows, mThreadsBlockCols);
	dim3 blockSizeDistance(mGridRows, mCols);
	int aTRows, aTCols, aGRows;
	// if ((mCols % 2) == 0) {
	// 	if (mCols > 16) {
	// 	}
	// }
	getBlockSize(mRows, mCols, 2, 0, aTRows, aTCols, aGRows);
	dim3 threadsPerBlockBMUDists(aTRows, aTCols);
	dim3 blockSizeBMUDists(aGRows, (mCols+1)/2);

	dim3 threadsPerBlockAdaptation(mThreadsBlockRows, mThreadsBlockCols);
	dim3 blockSizeAdaptation(mGridRows, mCols);

	int weightSizeOf = mWeightSize*sizeof(double);
	int distanceSizeOf = getMapSize()*sizeof(double);
	int sharedDistanceSizeOf = mThreadsBlockRows*mThreadsBlockCols*sizeof(double);
	cout << "Total shared memory needed size: " << sharedDistanceSizeOf << endl;

	int adapSizeOf = mThreadsBlockRows*sizeof(double);

	double *input_host = (double *) malloc(weightSizeOf);
	double *distance_host = (double *) malloc(distanceSizeOf);
	
	double *input_device, *distance_device, *bmu_device;
	cudaMalloc((void **) &input_device, weightSizeOf);
	cudaMalloc((void **) &distance_device, distanceSizeOf);
	cudaMalloc((void **) &bmu_device, weightSizeOf);
	
	cout << "Mapsize: " << getMapSize() << endl;
	cout << "Full map/neurons size: " << mNeuronsSize << endl; 
	cout << "Number of epochs: " << mEpoch << endl;
	cout << "Number of training set elements: " << mTrainingSetLength << endl;
	cout << "Learning rate: " << mLearningRate << endl;

	cudaEvent_t start, stop;
	float timer;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	for (int e = 0; e < mEpoch; ++e) {
		double currentRadius = neighborhoodRadiusDecay(mRadius, e);
		double radiusSquare = currentRadius*currentRadius;
		double decay = learningDecay(mLearningRate, e, mEpoch);

		if (show) {
			cudaMemcpy(mMap_host, mMap_device, mMapSizeOf, cudaMemcpyDeviceToHost);
			showColorMap(cellSize, 33);
		}

		// Comment me if not needed
		cout << "Epoch: " << e+1 << "/" << mEpoch << endl;
		cout << "Current radius: " << currentRadius << " - Square: " << radiusSquare << " - Learning Decay: " << decay << endl;

		for (int i = 0; i < mTrainingSetLength; ++i) {
			cudaMemset(distance_device, 0, distanceSizeOf);
			cudaMemset(bmu_device, 0, weightSizeOf);

			memset(distance_host, 0, distanceSizeOf);
			memset(input_host, 0, weightSizeOf);
			memcpy(input_host, &mTrainingSet[i * mWeightSize], weightSizeOf);

			// Solo debug
			// cout << endl;
			// cout << "Input" << endl;
			// for (int p = 0; p < mWeightSize; ++p) {
			// 	cout << input_host[p] << "\t";
			// } cout << endl;

			cudaMemcpy(input_device, input_host, weightSizeOf, cudaMemcpyHostToDevice);
			findBMU<<<blockSizeDistance, threadsPerBlockDistance, sharedDistanceSizeOf>>>(distance_device, input_device, mMap_device, mRows, mCols, mWeightSize);		
			
			// cerr << cudaGetErrorString(cudaGetLastError()) << endl;
			// Solo debug
			// cudaMemcpy(distance_host, distance_device, distanceSizeOf, cudaMemcpyDeviceToHost);
			// cout << endl;
			// cout << "Distance" << endl;
			// for (int p = 0; p < getMapSize(); ++p) {
			// 	cout << p+1 << ": " << distance_host[p] << "\n";
			// } cout << endl;

			if (cublasIdamin(mCublasHandler, getMapSize(), distance_device, 1, &minDist) != CUBLAS_STATUS_SUCCESS) {
				cout << "\nMin failed" << endl;
			} else {
				int bmuPos = minDist - 1; // Indexing starts from 1;
				// cout << "\nMin " << bmuPos << endl;

				int bmuColIdx = bmuPos % mCols; // Leave n, do not use z*n;
				int bmuRowIdx = (bmuPos - bmuColIdx) / mCols;  

			 	// bmuPos = bmuRowIdx * mCols * mWeightSize + bmuColIdx * mWeightSize;

			 	// Solo debug
				// cout << "\nCol: " << bmuColIdx << endl;
				// cout << "\nRow: " << bmuRowIdx << endl;
				// cout << "\nPos: " << bmuPos << endl;

				cudaMemset(distance_device, 9999, distanceSizeOf);
				cudaMemcpy(bmu_device, &mMap_device[bmuPos], weightSizeOf, cudaMemcpyDeviceToDevice);

				// Solo debug
				// 	cudaMemcpy(input_host, input_device, weightSizeOf, cudaMemcpyDeviceToHost);
				// 	cout << endl;
				// 	cout << "BMU" << endl;
				// 	for (int p = 0; p < mWeightSize; ++p) {
				// 		cout << input_host[p] << "\t";
				// 	} cout << endl;

				findBMUDistances<<<blockSizeBMUDists, threadsPerBlockBMUDists>>>(bmuPos, bmuColIdx, bmuRowIdx, mMap_device, mRows, mCols, mWeightSize, distance_device);

				// cudaMemcpy(distance_host, distance_device, distanceSizeOf, cudaMemcpyDeviceToHost);
				// cout << "Distance adaptation" << endl;
				// for (int p = 0; p < getMapSize(); ++p) {
				// 	cout << distance_host[p] << "\t";
				// } cout << endl;

				adaptation<<<blockSizeDistance, threadsPerBlockDistance>>>(bmuPos, mRows, mCols, mWeightSize, mMap_device, input_device, distance_device, radiusSquare, e, mEpoch, mLearningRate, decay);
			}
		}		
	}	
	cudaMemcpy(mMap_host, mMap_device, mMapSizeOf, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventElapsedTime(&timer, start, stop);
	cout << "Time duration: " << timer << "ms" << endl;

	free(input_host);
	free(distance_host);

	cudaFree(input_device);
	cudaFree(distance_device);
	cudaFree(bmu_device);

	cout << "Training finished" << endl;
	inputQuantizationError();

	if (show) {
		showColorMap(cellSize, 0);
	}
}

int cuSOM::getBMU() {
	return -1;
}

void cuSOM::evaluate(double *input_host, int &bmu_number, int &bmu_map_position, int *bmu_row_col_position) {
	cout << "Evaluating" << endl;

	cout << endl << "Input to test" << endl;
	for (int p = 0; p < mWeightSize; ++p) {
		cout << input_host[p] << "\t";
	} cout << endl << endl;

	int minDist;
	dim3 threadsPerBlockDistance(mThreadsBlockRows, mThreadsBlockCols);
	dim3 blockSizeDistance(mGridRows, mCols);

	int weightSizeOf = mWeightSize*sizeof(double);
	int distanceSizeOf = getMapSize()*sizeof(double);
	int sharedDistanceSizeOf = mThreadsBlockRows*mThreadsBlockCols*sizeof(double);
	cout << "total sharedmemory needed size: " << sharedDistanceSizeOf << endl;

	double *input_device, *distance_device;

	cudaMalloc((void **) &input_device, weightSizeOf);
	cudaMalloc((void **) &distance_device, distanceSizeOf);

	cudaMemset(distance_device, 0, distanceSizeOf);

	cudaMemcpy(input_device, input_host, mWeightSize, cudaMemcpyHostToDevice);
	findBMU<<<blockSizeDistance, threadsPerBlockDistance, sharedDistanceSizeOf>>>(distance_device, input_device, mMap_device, mRows, mCols, mWeightSize);

	if (cublasIdamin(mCublasHandler, getMapSize(), distance_device, 1, &minDist) != CUBLAS_STATUS_SUCCESS) {
		cout << "\nMin failed" << endl;
	} else {
		int bmuPos = minDist - 1; // Indexing starts from 1;
		// printf("\nMin %d\n", minDist);

		int bmuColIdx = bmuPos % mCols;
		int bmuRowIdx = (bmuPos - bmuColIdx) / mCols;  

		bmu_number = bmuPos;
		bmu_row_col_position[0] = bmuColIdx;
		bmu_row_col_position[1] = bmuRowIdx;
		bmu_map_position = bmuRowIdx * mCols * mWeightSize + bmuColIdx * mWeightSize;
	}
	cudaFree(input_device);
	cudaFree(distance_device);
}

void cuSOM::printMap() {
	cout << "\nMap\n" << endl;
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

void cuSOM::printMapDifferences() {
	cout << "\nMap differences\n" << endl;
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

void cuSOM::printTrainingSet() {
	cout << "\nTrainingSet\n" << endl;

	for (int i = 0; i < mTrainingSetLength; ++i) {
		cout << "Sample " << (i+1) << endl;
		for (int j = 0; j < mWeightSize; ++j) {
			cout << mTrainingSet[i * mWeightSize + j] << "\t";
		}
		cout << endl << endl;
	} 
}

void cuSOM::printMatlabMap() {
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

int cuSOM::getMapSize() {
	return mRows*mCols;
}

int cuSOM::getMapNeuronSize() {
	return mNeuronsSize;
}

void cuSOM::setTrainingSet(double *trainingSet, int length) {
	mTrainingSet = (double *) malloc(length*mWeightSize*sizeof(double));
	memcpy(mTrainingSet, trainingSet, length*mWeightSize*sizeof(double));
	mTrainingSetLength = length;
}

void cuSOM::initMap(int type, int device) {
	mType = type;
	cudaMalloc((void **) &mMap_device, mMapSizeOf);
	cudaMemset(mMap_device, 0, mMapSizeOf);
	generateUniformRandomArray(mMap_host, mNeuronsSize);
	cudaMemcpy(mMap_device, mMap_host, mMapSizeOf, cudaMemcpyHostToDevice);
	getBlockSize(mRows, mCols, mWeightSize, device, mThreadsBlockRows, mThreadsBlockCols, mGridRows);
	cout << "block size" << endl;
}

double cuSOM::neighborhoodRadiusDecay(double sigma, int epoch) {
	double lambda = ((double) mEpoch)/log(sigma);
	double e = exp(-((double) epoch / lambda));
	return sigma * e;
}

void cuSOM::showColorMap(int cellSize, int wait, string name) {
	if (mWeightSize > 3) {
		cout << "More than 3 values" << endl;
	} else {
		Mat colorMap(mRows*cellSize, mCols*cellSize, CV_32FC3);
			for (int i = 0; i < mRows; ++i) {
				int startX = i * cellSize;
				for (int j = 0; j < mCols; ++j) {
					int startY = j * cellSize; 
					int rIdx = i * mCols * mWeightSize + (j * mWeightSize);
					int gIdx = i * mCols * mWeightSize + (j * mWeightSize) + 1;
					int bIdx = i * mCols * mWeightSize + (j * mWeightSize) + 2; 
					rectangle(colorMap, Point(startX, startY), Point(startX + cellSize, startY + cellSize), Scalar(mMap_host[bIdx], mMap_host[gIdx], mMap_host[rIdx]), CV_FILLED);
				}
			}
		imshow("SOM Color Map: " + name, colorMap);
		waitKey(wait);
	}
}

void cuSOM::inputTopographicError() {

}

void cuSOM::inputQuantizationError() {
	int minDist;

	dim3 threadsPerBlockDistance(mThreadsBlockRows, mThreadsBlockCols); // Il numero di thread dipende solo dalla grandezza dell'input
	dim3 blockSizeDistance(mGridRows, mCols); 

	int weightSizeOf = mWeightSize*sizeof(double);
	int distanceSizeOf = getMapSize()*sizeof(double);
	int sharedDistanceSizeOf = mThreadsBlockRows*mThreadsBlockCols*sizeof(double);

	double *input_device, *distance_device;
	cudaMalloc((void **) &input_device, weightSizeOf);
	cudaMalloc((void **) &distance_device, distanceSizeOf);
	
	double *error_device;
	cudaMalloc((void **) &error_device, mTrainingSetLength*sizeof(double));

	// Use codebook to avoid other computations
	for (int i = 0; i < mTrainingSetLength; ++i) {
		cudaMemset(input_device, 0, distanceSizeOf);
		cudaMemset(distance_device, 0, distanceSizeOf);
	
		cudaMemcpy(input_device, &mTrainingSet[i * mWeightSize], weightSizeOf, cudaMemcpyHostToDevice);

		findBMU<<<blockSizeDistance, threadsPerBlockDistance, sharedDistanceSizeOf>>>(distance_device, input_device, mMap_device, mRows, mCols, mWeightSize);
		if (cublasIdamin(mCublasHandler, getMapSize(), distance_device, 1, &minDist) != CUBLAS_STATUS_SUCCESS) {
				cout << "\nMin failed" << endl;
		} else {
			int bmuPos = minDist - 1;
			cudaMemcpy(&error_device[i], &distance_device[bmuPos], sizeof(double), cudaMemcpyDeviceToDevice);
		}
	}
	double error_host;
	if (cublasDasum(mCublasHandler, mTrainingSetLength, error_device, 1, &error_host) != CUBLAS_STATUS_SUCCESS) {
		cout << "\nSum failed" << endl;
	} else {
		cout << "Quantization error: " << error_host << endl;
		cout << "Quantization error average: " << (error_host/mTrainingSetLength) << endl;
	}
}

void cuSOM::quantizationError(double * input) {

}

cuSOM::~cuSOM() {
	free(mMap_host);
	free(mTrainingSet);
	cudaFree(mMap_device);
}