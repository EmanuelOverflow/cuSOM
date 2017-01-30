#include "main.hpp"

void generateUniformRandomArray(double *input, int size) {
  	for (int i = 0; i < size; ++i) {
  		input[i] = rand() / (RAND_MAX + 1.0);
  	}
}

void readCSVDataset(int numInputs, int weightSize, double *trainingSet) {
	ifstream file ( "dataset/boston/input.in" );
	string value;
	//for (i = 0; getline(file, value); ++i) {
    for (int i = 0; i < numInputs; ++i) {
    	getline (file, value);
    	//cout << value << endl;
    	for (int j = 0; j < weightSize; ++j) {
    		int pos = value.find(" ");
		    string token = value.substr(0, pos);
		    //cout << token << "-" << token.length() << endl;
		    trainingSet[i * weightSize + j] = atof(token.c_str());
		    value.erase(0, pos + 1);
		}
    }
}

// void showColorInput(double *input) {
// 	Mat inputMat(100, 100, CV_32FC3, Scalar(input[2], input[1], input[0]));
// 	imshow("Input", inputMat);
// 	waitKey(0);
// }

int main(int argc, char *argv[]) {

	cmdline::parser cmd;
	cmd.add<int>("rows", 'r', "rows number", true);
	cmd.add<int>("cols", 'c', "cols number", true);
	cmd.add<int>("weight", 'w', "weight size", true);
	cmd.add<int>("num-inputs", 'i', "number of inputs", false, 10);
	cmd.add<int>("epoch", 'e', "number of epoch", false, 1000);
	cmd.add<double>("lr", 'l', "learning rate", false, 0.05);
	cmd.add("train-color-map", 0, "Show color map evolution during training");
	cmd.add("help", 0, "--rows, --cols, --weight are mandatory");
	
	if (!cmd.parse(argc, argv)) {
		cerr  << cmd.error() << endl << cmd.usage();
		return -1;
	}

	srand((unsigned) time(NULL));

	int rows = cmd.get<int>("rows");
	int cols = cmd.get<int>("cols");
	int weightSize = cmd.get<int>("weight");
	int numInputs = cmd.get<int>("num-inputs");
	int epoch = cmd.get<int>("epoch");
	double learningRate = cmd.get<double>("lr");

	bool showTrainingColorMap = cmd.exist("train-color-map");

	// if (argc > 0) {
	// 	rows = atoi(argv[1]);
	// 	cols = atoi(argv[2]);
	// 	weightSize = atoi(argv[3]);
	// 	numInputs = atoi(argv[4]);
	// 	epoch = atoi(argv[5]);
	// 	learningRate = atof(argv[6]);
	// } else {
	// 	cout << "Error no inputs" << endl;
	// }

	int inputsSizeOf = numInputs*weightSize*sizeof(double);
	int inputsSize = numInputs*weightSize;
	double *trainingSet = (double *) malloc(inputsSizeOf);
	generateUniformRandomArray(trainingSet, inputsSize);
	//readCSVDataset(numInputs, weightSize, trainingSet);

	serialSOM SOM(rows, cols, weightSize, epoch, learningRate);
	SOM.setTrainingSet(trainingSet, numInputs);
	// SOM.printTrainingSet();
	// SOM.printMatlabMap();
	// SOM.printMap();
	// SOM.showColorMap(5);
	SOM.train(showTrainingColorMap); 
	// SOM.printMatlabMap();
	// SOM.printMap();
	// SOM.showColorMap(5);
	
	// SOM.printMapDifferences();




	// int inputSize = weightSize*sizeof(double);
	// double *input = (double *) malloc(inputSize);
	// generateUniformRandomArray(input, weightSize);
	// int bmu_number = -1;
	// int bmu_map_position = -1;
	// int *bmu_row_col_position = (int *) malloc(2*sizeof(int));

	// SOM.evaluate(input, bmu_number, bmu_map_position, bmu_row_col_position);
	// cout << "BMU number: " << bmu_number << endl;
	// cout << "BMU position on map: " << bmu_map_position << endl;

	// showColorInput(input);
	// SOM.showColorMap();
}