#include "../header/global.cuh"

__host__ __device__ double learningDecay(double learningRate, int currentEpoch, int totalEpoch) {
	return learningRate * exp(- (double) currentEpoch / totalEpoch);
}

__host__ __device__ double neighborhoodFunction(double distance, int radiusSquare, double learningRate) {
	return exp(-(double) (distance) / (2*radiusSquare));
}

__global__ void findBMU(double *out, double *vec, double *mtx, int m, int n, int z) {
	
	extern __shared__ double dShared[];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int mtxRowIdx = tx + blockDim.x * bx;
	int rowIdx = tx;
	int colIdx = ty + z * by;
	
	int dimIdx = ty;

	if (dimIdx < z && mtxRowIdx < m) {
		dShared[rowIdx * blockDim.y + dimIdx] = mtx[mtxRowIdx * n * z + colIdx] - vec[dimIdx]; // 1 access to global memry
		dShared[rowIdx * blockDim.y + dimIdx] *= dShared[rowIdx * blockDim.y + dimIdx]; // 1 access to shared memory
	} else {
		dShared[rowIdx * blockDim.y + dimIdx] = 0.0;
	}

	__syncthreads();

	// Parallel reduction 
	for (int i = blockDim.y/2; i > 0; i >>= 1) {
		if (dimIdx < i) {
			dShared[rowIdx * blockDim.y + dimIdx] += dShared[i + (rowIdx * blockDim.y + dimIdx)];
		}
		__syncthreads();
	}

	if (dimIdx == 0 && mtxRowIdx < m) {
		out[mtxRowIdx * n + by] = sqrt(dShared[rowIdx * blockDim.y]);
	}
}

__global__ void findBMUDistances(int bmuPos, int bmuColIdx, int bmuRowIdx, double *mat, int m, int n, int z, double *dist) {
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int rowIdx = tx + blockDim.x * bx;
	int colIdx = ty + blockDim.y * by;

	int neighborPos = rowIdx * n + colIdx;
                        
	if (rowIdx < m && colIdx < n) {
		dist[neighborPos] = pow((double) (rowIdx - bmuRowIdx), 2) + pow((double) (colIdx - bmuColIdx), 2);		
	}
}

__global__ void adaptation(int bmuPos, int m, int n, int z, double *map, double *input, double *distance, double radiusSquare, int currentEpoch, int totalEpoch, double learningRate, double decay) {
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int mtxRowIdx = tx + blockDim.x * bx;
	if (mtxRowIdx < m && ty < z) {
		int colIdx = ty + z * by;

		int position = mtxRowIdx * n * z + colIdx; 
		int distIdx = mtxRowIdx * n + by; 

		if (distance[distIdx] < radiusSquare) {
	    	double influence = neighborhoodFunction(distance[distIdx], radiusSquare, learningRate);
	        map[position] += influence * decay * (input[ty] - map[position]);
	    }
	}
}

__global__ void adaptationSM(int bmuPos, int m, int n, int z, double *map, double *input, double *distance, double radiusSquare, int currentEpoch, int totalEpoch, double learningRate, double decay) {

	extern __shared__ double dist[];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int mtxRowIdx = tx + blockDim.x * bx;
	if (mtxRowIdx < m && ty < z) {
		int colIdx = ty + z * by;

		int position = mtxRowIdx * n * z + colIdx; 
		int distIdx = mtxRowIdx * n + by;

		if (ty == 0) {
			dist[tx] = distance[distIdx];
		}
		
		__syncthreads();
		
		if (dist[tx] < radiusSquare) {
		    	double influence = neighborhoodFunction(dist[tx], radiusSquare, learningRate);
		        map[position] += influence * decay * (input[ty] - map[position]);
	    }
	}
}

__host__ bool isPowerOfTwo (unsigned int x) {
	return ((x != 0) && ((x & (~x + 1)) == x));
}

__host__ int getNextPow2(int value) {
	if (isPowerOfTwo((unsigned int) value)) {
		return value;
	} else {
		return pow(2, (int) (log2(value - 1.0) + 1));
	}
}

__host__ void getBlockSize(int row_size, int col_size, int weight_size, int device, int &out_thread_rows, int &out_thread_weight, int &out_grid_row) {
	struct cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, device);
	int max_threads = properties.maxThreadsPerBlock;
	
	out_thread_weight = getNextPow2(weight_size);
	int maxRows = max_threads / out_thread_weight;
	int minRows = getNextPow2(min(row_size, maxRows));
	out_thread_rows = out_thread_weight < minRows ? (minRows / out_thread_weight) : minRows;
	out_grid_row = getNextPow2(row_size) / out_thread_rows; // sempre uguale ad out_thread_weight così
	// int p = 0;
	// for (int i = 1; p < out_thread_rows; ++i) {
	// 	p = pow(2, i);
	// 	out_grid_row = out_thread_rows / p;
	// 	if ((out_grid_row*col_size) <= 16) {
	// 		break;
	// 	}
	// }
	// out_thread_rows /= out_grid_row;
	
	cout << endl;
	cout << "maxThreadsDim x: " << properties.maxThreadsDim[0] << endl;
	cout << "maxThreadsDim y: " << properties.maxThreadsDim[1] << endl;
	cout << "maxThreadsDim z: " << properties.maxThreadsDim[2] << endl;
	cout << "maxThreadsPerBlock: " << properties.maxThreadsPerBlock << endl;
	cout << endl;
	cout << "totalGlobalMem: " << properties.totalGlobalMem << endl;
	cout << "sharedMemPerBlock: " << properties.sharedMemPerBlock << endl;
	cout << "sharedMemPerMultiprocessor: " << properties.sharedMemPerMultiprocessor << endl;
	cout << endl;
	cout << "Number of rows in a block: " << out_thread_rows << endl;
	cout << "Number of weights in a block: " << out_thread_weight << endl;
	cout << "Total number of threads: " << (out_thread_rows*out_thread_weight) << endl;
	cout << "Number of rows in the grid: " << out_grid_row << endl;
	cout << "Number of cols in the grid: " << col_size << endl;
	cout << "Grid size: " << (out_grid_row*col_size) << endl;
	cout << endl;
}

__host__ void generateUniformRandomArray(double *input, int size) {
  	for (int i = 0; i < size; ++i) {
  		input[i] = rand() / (RAND_MAX + 1.0);
  	}
}