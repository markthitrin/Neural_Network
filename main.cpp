#include <initializer_list>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "Neural_Network.h"
#include <chrono>
#include <vector>
#include <thread>

const int DATASET_SIZE = 4;

std::vector<std::vector<double>> output_to_loss;

std::vector<double> global_dataset[DATASET_SIZE];

void train_fix(std::vector<double> (&dataset)[DATASET_SIZE], Neural_Network& NN, const Matrix<double>& power_weight, const Matrix<double>& power_bias) {
	NN.set_change_dependencies(0);

	NN.set_weight(0, Power::weight_type::POWER, power_weight);
	NN.set_bias(0, Power::bias_type::BIAS, power_bias);
	std::cout << "START" << std::endl;
	NN.print_weight();
	for (int i = 0; i < 200; i++) {
		for (int j = 0; j < dataset[0].size(); j++) {
			Matrix<double> input({ {dataset[0][j]}, {dataset[1][j]} });
			Matrix<double> target({ {dataset[2][j]} , {dataset[3][j]} });

			NN.feedforward(input);
			NN.backpropagation(target);

			if (NN.get_max_abs_change_dependencies() > 0.2) {
				NN.mul_change_dependencies(double(0.2) / NN.get_max_abs_change_dependencies());
			}
			NN.change_dependencies();
			NN.set_change_dependencies(0);
		}
	}
	std::cout << "END\n";
	NN.print_weight();
}

double get_avarage_loss(std::vector<double>(&dataset)[DATASET_SIZE], Neural_Network& NN) {
	double sum = 0;
	for (int i = 0; i < dataset[0].size(); i++) {
		Matrix<double> input({ {dataset[0][i]}, {dataset[1][i]}});
		Matrix<double> target({ {dataset[2][i]} , {dataset[3][i]}});

		NN.feedforward(input);

		sum += NN.get_loss(target);
	}
	sum /= 500;
	return sum;
}

void create_dataset(std::vector<double> (&dataset)[DATASET_SIZE], int number) {
	/*for (int i = 0; i < number; i++) {
		double a = double(rand() % 100) / 10;
		double b = double(rand() % 100) / 10;*/
	for (int a = 0; a < 20; a++) {
		for (int b = 0; b < 20; b++) {
		double factor1 = 1.375;
		double factor2 = 2.125;
		dataset[0].push_back(a);
		dataset[1].push_back(b);
		dataset[2].push_back(std::pow(a,factor1));
		dataset[3].push_back(std::pow(b,factor2));
		}
	}
	/*}*/
}

void copy_dataset(std::vector<double>(&dataset)[DATASET_SIZE]) {
	for (int i = 0; i < DATASET_SIZE; i++) {
		for (int j = 0; j < global_dataset[i].size(); j++) {
			dataset[i].push_back(global_dataset[i][j]);
		}
	}
}

void train(double offset_i, double offset_j) {
	std::vector<double> dataset[DATASET_SIZE];
	
	copy_dataset(dataset);

	std::vector<LayerId> model = {
		LayerId(Layer::POWER, 2, "act:linear dact:dlinear"),
		LayerId(Layer::DENSE, 2)
	};
	Neural_Network AI(model, mean_squre_loss_func, dmean_squre_loss_func);
	AI.set_learning_rate(0.001);
	AI.rand_weight({ {0.0,0.0} });
	AI.rand_bias({ {0.0,0.0} }); 

	for (double i = offset_i; i < 5.0; i += 0.25) {
		for (double j = offset_j; j < 5.0; j += 0.25) {
			std::cout << i << " " << j << std::endl;
			Matrix<double> power_weight({ {i} , {j} });
			Matrix<double> bias_weight({ {0}, {0} });

			train_fix(dataset, AI, power_weight, bias_weight);

			double loss = get_avarage_loss(dataset, AI);
			output_to_loss.push_back(std::vector<double>({ i, j, loss }));
		}
	}
}

void print_loss() {
	std::ofstream loss("Output/loss.txt");

	for (int i = 0; i < output_to_loss.size(); i++) {
		for (int j = 0; j < output_to_loss[i].size(); j++) {
			loss << output_to_loss[i][j] << " ";
		}
		loss << "\n";
	}
}

int main() {

	srand(time(0));
	create_dataset(global_dataset, 500);
	std::srand(std::time(0));

	std::thread t1(train,0,0);
	//std::thread t2(train, 0.125, 0);
	//std::thread t3(train, 0, 0.125);
	//std::thread t4(train, 0.125, 0.125);
	
	t1.join();
	//t2.join();
	//t3.join();
	//t4.join();

	print_loss();
	std::cout << "train succesfully\n";


	/*while (true) {
		double input1, input2;
		std::cin >> input1 >> input2;
		Matrix<double> input({ {input1},{input2} });
		AI.feedforward(input);
		std::cout << AI.get_output()[0][0] << '\n';
	}*/
}