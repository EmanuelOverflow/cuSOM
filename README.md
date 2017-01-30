# CUSOM

CuSOM is a CUDA implementation of [Self Organizing Map](https://en.wikipedia.org/wiki/Self-organizing_map).

### Important
It was developed for academic purpose, some features are not available or need to be modified.

## Prerequisites

- Cuda 7.x;
- Cublas;
- OpenCV 2.4;

## Content

There are two versions of SOM:

- batchSOM (standalone):
	* Compiles main.cpp, serialSOM;
- Cuda SOM:
	* Compiles *.cu files;


## Compilation

nvcc -G -g main.cu cuSOM.cu global.cu -o SOMMM -lcurand -lcublas `pkg-config --cflags --libs opencv`
