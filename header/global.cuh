#ifndef GLOBAL_CUH
#define GLOBAL_CUH

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>


using namespace std;

__host__ __device__ double learningDecay(double, int, int);
__host__ __device__ double neighborhoodFunction(double, int, double);
__global__ void adaptation(int, int, int, int, double *, double *, double *, double, int, int, double, double);
__global__ void adaptationSM(int, int, int, int, double *, double *, double *, double, int, int, double, double); // extern
__global__ void findBMUDistances(int, int, int, double*, int, int, int, double *);
__global__ void findBMU(double *, double *, double *, int, int, int); // extern

__host__ void getBlockSize(int, int, int, int, int &, int &, int &);
__host__ int getNextPow2(int);
__host__ bool isPowerOfTwo (unsigned int);
__host__ void generateUniformRandomArray(double *, int);

#endif