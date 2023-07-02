#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <arm_neon.h>
#include <time.h>
#include <cstdlib>
#include <fstream>
#include <pthread.h>
#include <arm_neon.h>
#include<assert.h>
#include <stdio.h>
#include<cmath>
#include <iostream>
#include<fstream>
#include<sstream>
#include<string>
#define INTERVAL 1000
using namespace std;
typedef long long ll;
const int ROWS = 15000;
const int COLS = 9;
const int dim = 9;
const int trainNum = 12000;
const int testNum = 3000;
float train[trainNum][dim];
float test[testNum][dim];
bool y_train[trainNum];
bool y_test[testNum];
//float dist[testNum][trainNum];
using namespace std;

const int threadcount = 6;

struct DataPoint
{
	vector<double> features;
	int label;
};

double euclidean_distance(const vector<double>& a, const vector<double>& b) {
	double distance = 0.0;
	//    cout<<"a.size: "<<a.size()<<endl;  //128
	//    cout<<"b.size: "<<b.size()<<endl;  //128
	for (size_t i = 0; i < a.size(); i++)
	{
		distance += pow(a[i] - b[i], 2);
	}
	return sqrt(distance);
}

int knn_predict(const vector<DataPoint>& training_set, const vector<double>& test_point, int k) {
	vector<pair<double, int>> distances(training_set.size());
	//    cout<<"training_set.size: "<<training_set.size()<<endl;  //10000
	for (size_t i = 0; i < training_set.size(); i++)
	{
		double distance = euclidean_distance(training_set[i].features, test_point);
		distances[i] = make_pair(distance, training_set[i].label);
	}
	sort(distances.begin(), distances.end());
	//    for (int i = 0; i < training_set.size(); i++)
	//    {
	//        cout<<distances[i].first<<endl;
	//    }
	//升序
	vector<int> k_nearest_neighbors(k);
	for (int i = 0; i < k; i++)
	{
		k_nearest_neighbors[i] = training_set[i].label;
	}
	//sort(k_nearest_neighbors.begin(), k_nearest_neighbors.end());
	int max_count = 1;
	int mode = k_nearest_neighbors[0];
	int count = 1;
	for (int i = 1; i < k; i++)
	{
		if (k_nearest_neighbors[i] == k_nearest_neighbors[i - 1])
		{
			count++;
		}
		else
		{
			if (count > max_count)
			{
				max_count = count;
				mode = k_nearest_neighbors[i - 1];
			}
			count = 1;
		}
	}
	if (count > max_count)
	{
		mode = k_nearest_neighbors[k - 1];
	}
	return mode;
}



double euclidean_distance_parallel(vector<double>& a, vector<double>& b)
{
	double distance = 0.0;
	float64x2_t a_reg, b_reg, d_reg, sum_reg = vdupq_n_f64(0.0);
	for (size_t i = 0; i < a.size(); i += 2)
	{
		a_reg = vld1q_f64(&a[i]);
		b_reg = vld1q_f64(&b[i]);
		d_reg = vsubq_f64(a_reg, b_reg);
		sum_reg = vmlaq_f64(sum_reg, d_reg, d_reg);
	}
	double sum[2];
	vst1q_f64(sum, sum_reg);
	for (int i = 0; i < 2; i++)
	{
		distance += sum[i];
	}
	for (size_t i = a.size() - (a.size() % 2); i < a.size(); i++)
	{
		distance += pow(a[i] - b[i], 2);
	}
	return sqrt(distance);
}

struct threaddata
{
	vector<DataPoint> *training_set;
	vector<double> *test_point;
	vector<pair<double, int>> *distances;
	int i;
	int num;
	threaddata() {}
	threaddata(vector<DataPoint> *training_set,
		vector<double> *test_point,
		vector<pair<double, int>> *distances,
		int i, int num)
	{
		this->set_values(training_set, test_point, distances, i, num);
	}
	void set_values(vector<DataPoint> *training_set,
		vector<double> *test_point,
		vector<pair<double, int>> *distances,
		int i, int num)
	{
		this->training_set = training_set;
		this->test_point = test_point;
		this->distances = distances;
		this->i = i;
		this->num = num;
	}
};

typedef struct threaddata ThreadData;

void* cal_dis_pthread(void* arg)
{
	ThreadData* data = static_cast<ThreadData*>(arg);
	size_t id = data->i;
	for (int i = id * data->num; i < (id + 1) * data->num && i < data->training_set->size(); i++)
	{
		double distance = euclidean_distance_parallel((*data->training_set)[i].features, *data->test_point);
		(*data->distances)[i] = make_pair(distance, (*data->training_set)[i].label);
	}
	return NULL;
}

int knn_predict_parallel(vector<DataPoint>& training_set, vector<double>& test_point, int k)
{
	vector<pair<double, int>> distances(training_set.size());
	pthread_t* tid;
	int per_num = training_set.size() / 4;
	tid = static_cast<pthread_t*>(malloc(threadcount * sizeof(pthread_t)));

	ThreadData data[threadcount];
	for (int i = 0; i < threadcount; i++)
	{
		data[i].set_values(&training_set, &test_point, &distances, i, per_num);
	}
	for (int i = 0; i < threadcount; i++)
	{
		pthread_create(&tid[i], NULL, cal_dis_pthread, static_cast<void*>(&data[i]));
	}

	for (int i = 0; i < threadcount; i++)
	{
		pthread_join(tid[i], NULL);
	}
	free(tid);

	sort(distances.begin(), distances.end());
	vector<int> k_nearest_neighbors(k);
	for (int i = 0; i < k; i++)
	{
		k_nearest_neighbors[i] = distances[i].second;
	}
	sort(k_nearest_neighbors.begin(), k_nearest_neighbors.end());
	int max_count = 1;
	int mode = k_nearest_neighbors[0];
	int count = 1;
	for (int i = 1; i < k; i++)
	{
		if (k_nearest_neighbors[i] == k_nearest_neighbors[i - 1])
		{
			count++;
		}
		else
		{
			if (count > max_count)
			{
				max_count = count;
				mode = k_nearest_neighbors[i - 1];
			}
			count = 1;
		}
	}
	if (count > max_count)
	{
		mode = k_nearest_neighbors[k - 1];
	}
	return mode;
}

void init_X()
{
	float myArray[ROWS][COLS]; // 定义二维数组

	// 读取CSV文件
	std::ifstream file("/home/ss2113881/data/data.csv");
	if (file.is_open()) {
		std::string line;
		int row = 0;
		while (std::getline(file, line) && row < ROWS) {
			std::istringstream ss(line);
			std::string cell;
			int col = 0;
			while (std::getline(ss, cell, ',') && col < COLS) {
				double value = std::stod(cell);
				myArray[row][col] = value;
				col++;
			}
			row++;
		}
		file.close();
	}
	else {
		std::cout << "Failed to open file." << std::endl;
	}

	// 将数据拷贝到test和train数组中
	for (int i = 0; i < testNum; i++) {
		for (int k = 0; k < dim; k++) {
			test[i][k] = myArray[i][k];
		}
	}

	for (int i = 0; i < trainNum; i++) {
		for (int k = 0; k < dim; k++) {
			train[i][k] = myArray[i + testNum][k];
		}
	}
}

void init_Y_train()
{
	// 读取CSV文件
	std::ifstream file("/home/ss2113881/data/y_train.csv");
	if (file.is_open()) {
		std::string line;
		int row = 0;
		while (std::getline(file, line) && row < trainNum) {
			std::istringstream ss(line);
			std::string cell;
			int col = 0;
			while (std::getline(ss, cell, ',') && col < 1) {
				int value = std::stoi(cell);  // 将单元格的值转换为int类型
				y_train[row] = value;
				col++;
			}
			row++;
		}
		file.close();
	}
	else {
		std::cout << "Failed to open file." << std::endl;
	}
}
void init_Y_test()
{
	// 读取CSV文件
	std::ifstream file("/home/ss2113881/data/y_test.csv");
	if (file.is_open()) {
		std::string line;
		int row = 0;
		while (std::getline(file, line) && row < testNum) {
			std::istringstream ss(line);
			std::string cell;
			int col = 0;
			while (std::getline(ss, cell, ',') && col < 1) {
				int value = std::stoi(cell);  // 将单元格的值转换为int类型
				y_test[row] = value;
				col++;
			}
			row++;
		}
		file.close();
	}
	else {
		std::cout << "Failed to open file." << std::endl;
	}
}

int main() {

	//test=X_test train=X_train
	int MAX_K = 30;
	int default_d = 9;
	int default_n = 12000;
	int default_k = MAX_K;
	int DIMENSIONS, n, k, predicted_label;
	//初始化导入X,从而导入特征向量
	init_X();
	init_Y_train();
	init_Y_test();

	srand(time(0));
	// k
	cout << "----------------Results under different K----------------" << endl;
	ofstream outfile_k("output_k_outcome_6.csv");
	n = default_n;
	DIMENSIONS = default_d;

	for (int k = 1; k <= MAX_K; k = k + 1)
	{
		cout << "Results with K: " << k << endl;
		//开始一轮的计时
		clock_t start_knn, end_knn;


		//初始化训练集
		vector<DataPoint> training_set(n);
		for (int i = 0; i < n; i++)
		{
			for (int d = 0; d < DIMENSIONS; d++)
			{
				training_set[i].features.push_back(train[i][d]);
			}
			training_set[i].label = y_train[i];
		}
		start_knn = clock();
		//初始化测试集，对每一个测试点依次计算
		for (int i = 0; i < testNum; i++) {
			DataPoint datapoint_test_point;
			for (int d = 0; d < DIMENSIONS; d++)
			{
				datapoint_test_point.features.push_back(test[i][d]);
			}
			vector<double> test_point;
			test_point = datapoint_test_point.features;
			//进行预测
			predicted_label = knn_predict(training_set, test_point, k);
			//cout << "Predicted label with kNN: " << predicted_label << endl;
		}
		end_knn = clock();   //结束时间
		double used_time = double(end_knn - start_knn);
		//double outcome = predict_precision / testNum;
		//cout << "Predicted precison with kNN: " << outcome << endl;
		cout << "Algorithm execution time: " << used_time << "ms" << endl;  //输出时间（单位：ms）





		//使用并行
		int predicted_label_parallel;
		clock_t start_parallel, end_parallel;
		start_parallel = clock();
		for (int i = 0; i < testNum; i++) {
			DataPoint datapoint_test_point;
			for (int d = 0; d < DIMENSIONS; d++)
			{
				datapoint_test_point.features.push_back(test[i][d]);
			}
			vector<double> test_point;
			test_point = datapoint_test_point.features;
			//进行预测
			predicted_label_parallel = knn_predict_parallel(training_set, test_point, k);
			//cout << "Predicted label with parallel_kNN: " << predicted_label_parallel << endl;
		}
		end_parallel = clock();
		double used_time_parallel = double(end_parallel - start_parallel);
		//double outcome_parallel = predict_precision_parallel / testNum;
		//cout << "predict precision parallel with parallel_kNN: " << outcome_parallel << endl;
		cout << "Parallel algorithm execution time: " << used_time_parallel << "ms" << endl;  //输出时间（单位:ms）

		cout << endl;
		outfile_k << k << "," << used_time << "," << used_time_parallel << endl;
		vector<DataPoint>().swap(training_set);
	}
	outfile_k.close(); // 关闭输出流


	return 0;
}



