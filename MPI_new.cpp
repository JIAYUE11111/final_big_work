#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>
#include <arm_neon.h>





float *dynamicArray(int m, int n) {
	float *mat = NULL;
	mat = (float*)calloc(m*n, sizeof(float));
	return mat;
}

float max(float *x, int n) {
	int i;
	float max = x[0];
	int maxIndex = 0;
	for (i = 0; i < n; i++) {
		if (x[i] >= max) {
			max = x[i];
			maxIndex = i;
		}
	}
	return(float)maxIndex;
}

void swap(float * a, float * b) {
	float t = *a;
	*a = *b;
	*b = t;
}

int partition(float arr[], int low, int high, float *y) {
	float pivot = arr[high]; // pivot
	int i = (low - 1);
	for (int j = low; j <= high - 1; j++) {
		if (arr[j] < pivot) {
			swap(&arr[i], &arr[j]);
			swap(&y[i], &y[j]);
		}
	}
	swap(&arr[i + 1], &arr[high]);
	swap(&y[i + 1], &y[high]);
	return (i + 1);
}
void quicksort(float arr[], int low, int high, float *y) {
	if (low < high) {
		int pi = partition(arr, low, high, y);
		quicksort(arr, low, pi - 1, y);
		quicksort(arr, pi + 1, high, y);
	}
}

void euclideanDistance(int data_size_process, float* distance_per_process, float* data_per_process, float* x, int NO_OF_FEATURES) {
	int index = 0;
	int simd_size = 4; // 假设使用NEON指令集，每次并行计算4个浮点数
	int simd_iterations = data_size_process / (simd_size * NO_OF_FEATURES);

	for (int i = 0; i < simd_iterations; i++) {
		float32x4_t distance_sum = vdupq_n_f32(0.0);

		for (int j = 0; j < NO_OF_FEATURES; j++) {
			float32x4_t data = vld1q_f32(data_per_process + (i * NO_OF_FEATURES + j) * simd_size);
			float32x4_t x_val = vdupq_n_f32(x[j]);

			float32x4_t diff = vsubq_f32(data, x_val);
			distance_sum = vmlaq_f32(distance_sum, diff, diff);
		}

		float32_t result[simd_size];
		vst1q_f32(result, distance_sum);

		for (int k = 0; k < simd_size; k++) {
			distance_per_process[index] = result[k];
			index++;
		}
	}
}

void classify_KNN(float *X_train, float *y_train, float *X_test, float *y_test, int rank, int size, int TRAINING_SIZE, int TESTING_SIZE, int NO_OF_FEATURES, int NO_OF_CLASSES, int n_neighbors) {
	int i, j, predictions, data_size_process, row_size_process;
	float *data_per_process, *overall_distance, *distance_per_process, *labels_per_process, *overall_labels;

	if (TRAINING_SIZE % size != 0) {
		if (rank == 0)
			printf("Number of rows in the training dataset should be divisibe by number of processors so that rows are divided equally.\n");
		MPI_Finalize();
		exit(0);
	}

	// initialise arrays
	row_size_process = TRAINING_SIZE / size;
	data_size_process = row_size_process * NO_OF_FEATURES;

	data_per_process = dynamicArray(data_size_process, 1);
	distance_per_process = dynamicArray(row_size_process, 1);
	overall_distance = dynamicArray(TRAINING_SIZE, 1);

	labels_per_process = dynamicArray(row_size_process, 1);
	overall_labels = dynamicArray(TRAINING_SIZE, 1);

	MPI_Scatter(X_train, data_size_process, MPI_FLOAT, data_per_process, data_size_process, MPI_FLOAT, 0, MPI_COMM_WORLD);

	float *x = dynamicArray(NO_OF_FEATURES, 1);
	float precision;
	int right_prediction = 0;
	for (i = 0; i < TESTING_SIZE; i = i + 1) {
		MPI_Scatter(y_train, row_size_process, MPI_FLOAT, labels_per_process, row_size_process, MPI_FLOAT, 0, MPI_COMM_WORLD);

		for (j = 0; j < NO_OF_FEATURES; j++)
			x[j] = X_test[i*NO_OF_FEATURES + j];

		euclideanDistance(data_size_process, distance_per_process, data_per_process, x, NO_OF_FEATURES);


		quicksort(distance_per_process, 0, row_size_process - 1, labels_per_process);

		MPI_Gather(distance_per_process, row_size_process, MPI_FLOAT, overall_distance, row_size_process, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Gather(labels_per_process, row_size_process, MPI_FLOAT, overall_labels, row_size_process, MPI_FLOAT, 0, MPI_COMM_WORLD);

		if (rank == 0) {
			quicksort(overall_distance, 0, TRAINING_SIZE - 1, overall_labels);
			float* neighborCount = dynamicArray(NO_OF_CLASSES, 1);
			int k;
			for (k = 0; k < n_neighbors; k++)
				neighborCount[(int)overall_labels[k]]++;

			predictions = (int)max(neighborCount, NO_OF_CLASSES);
			//	free(neighborCount);
			printf("%d) Prediction: %d   Original: %d\n\n", i, predictions, (int)y_test[i]);
			if (predictions == y_test[i])
			{
				right_prediction++;
			}
		}
		
	}
	precision = right_prediction / TESTING_SIZE;
	cout << "The precision is:" << precision << endl;
	//	free(x);
	//	free(overall_distance);
	//	free(distance_per_process);
}

int main() {
	const char *X_train_path = "/home/ss2113881/data/X_train.csv";
	const char*y_train_path = "/home/ss2113881/data/y_train.csv";
	const char*X_test_path = "/home/ss2113881/data/X_test.csv";
	const char *y_test_path = "/home/ss2113881/data/y_test.csv";
	int TRAINING_SIZE = 11999;
	int TESTING_SIZE = 3000;
	int NO_OF_FEATURES = 9;
	int NO_OF_CLASSES = 2;
	int n_neighbors = 1;
	float *X_train, *y_train, *X_test, *y_test;
	double t1, t2;
	int size, rank, index = 0;
	FILE *f;
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	if (rank == 0) {
		//X_train
		index = 0;
		f = NULL;
		X_train = NULL;
		X_train = dynamicArray(TRAINING_SIZE, NO_OF_FEATURES);
		f = fopen(X_train_path, "r");
		if (f == NULL) {
			printf("Error while reading the file\n");
			exit(1);
		}
		while (fscanf(f, "%f%*c", &X_train[index]) == 1) //%*c ignores the comma while reading the CSV
			index++;
		fclose(f);
		//Y_train
		index = 0;
		f = NULL;
		y_train = NULL;
		y_train = dynamicArray(TRAINING_SIZE, 1);
		f = fopen(y_train_path, "r");
		if (f == NULL) {
			printf("Error while reading the file\n");
			exit(1);
		}
		while (fscanf(f, "%f%*c", &y_train[index]) == 1)
			index++;
		fclose(f);
		y_train = y_train;
	}

	//	X_test 
	index = 0;
	f = NULL;
	X_test = NULL;
	X_test = dynamicArray(TRAINING_SIZE, NO_OF_FEATURES);
	f = fopen(X_test_path, "r");
	if (f == NULL) {
		printf("Error while reading the file\n");
		exit(1);
	}
	while (fscanf(f, "%f%*c", &X_test[index]) == 1)
		index++;
	fclose(f);
	X_test = X_test;


	//y_test
	index = 0;
	f = NULL;
	y_test = NULL;
	y_test = dynamicArray(TRAINING_SIZE, 1);
	f = fopen(y_test_path, "r");
	if (f == NULL) {
		printf("Error while reading the file\n");
		exit(1);
	}
	while (fscanf(f, "%f%*c", &y_test[index]) == 1)
		index++;
	fclose(f);
	y_test = y_test;

	if (rank == 0) {
		t1 = MPI_Wtime();
	}

	classify_KNN(X_train, y_train, X_test, y_test, rank, size, TRAINING_SIZE, TESTING_SIZE, NO_OF_FEATURES, NO_OF_CLASSES, n_neighbors);

	if (rank == 0) {
		t2 = MPI_Wtime();
	}

	if (rank == 0) {
		printf("Time  (%d Processors): %f\n", size, t2 - t1);
		//	free(X_train);
		//	free(y_train);
	}
	//free(X_test);
	//free(y_test);
	MPI_Finalize();
	return 0;
}
