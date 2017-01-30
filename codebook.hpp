#include <iostream>
#include <cstdlib>
#include <vector>

using namespace std;

class Codebook {
	public:
		Codebook() {};
		Codebook(int);
		void setTrainingSet(double *, int, int);
		void setTrainingSet(double *, int);
		void getTrainingSet(double *);
		void setDistance(double *, int);
		void getDistance(double *, int);
		void setBMU(double *, int, int);
		void getBMU(double *, int, int);
		void setMappedBMU(int *);
		void getMappedBMU(int *, int);

	private:
		vector<double *> mTrainingSet;
		vector<double> mDistances;
		vector<int> mBMU;
		vector<int *> mMappedBMU;

		int mWeightSize;
		int mWeightSizeOf;

		// oppure

		// double *mTrainingSet;
		// double *mDistances;
		// double *mBMUs;
		// double *mMappedBMUs;
};