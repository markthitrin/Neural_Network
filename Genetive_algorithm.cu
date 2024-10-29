#pragma once

#include "Header.cuh"
#include "Neural_Network.cu"

class Genetive_algorithm {
public:
	Genetive_algorithm() {

	}

	Genetive_algorithm(std::vector<LayerId> _layer, const std::function<double(Neural_Network&)> _simulator,
		const std::function<double(double)> _probability_func,
		const std::size_t& _population, const double& _mutation_chance,const double& _mutation_rate) :
	population(_population) , simulator(_simulator), probability_func(_probability_func),
	mutation_chance(_mutation_chance), mutation_rate(_mutation_rate) {
		for (int i = 0; i < population; i++) {
			AI.push_back(_layer);
			point.push_back(0);
		}	
	}


	void rand_weight(const std::vector<std::pair<double, double>>& setting) {
		for (int i = 0; i < AI.size(); i++) {
			AI[i].rand_weight(setting);
		}
	}

	void rand_weight(const std::vector<std::function<double()>>& setting) {
		for (int i = 0; i < AI.size(); i++) {
			AI[i].rand_weight(setting);
		}
	}

	void rand_weight(const std::vector<std::function<double(std::size_t, std::size_t)>>& setting) {
		for (int i = 0; i < AI.size(); i++) {
			AI[i].rand_weight(setting);
		}
	}

	void rand_bias(const std::vector<std::pair<double, double>>& setting) {
		for (int i = 0; i < AI.size(); i++) {
			AI[i].rand_bias(setting);
		}
	}

	void rand_bias(const std::vector<std::function<double()>>& setting) {

		for (int i = 1; i < AI.size(); i++) {
			AI[i].rand_bias(setting);
		}
	}

	void rand_bias(const std::vector<std::function<double(std::size_t, std::size_t)>>& setting) {
		for (int i = 0; i < AI.size(); i++) {
			AI[i].rand_bias(setting);
		}
	}



	void simulate() {
		for (int i = 0; i < AI.size(); i++) {
			std::cout << "simulalte ; " << i << " / " << AI.size() << std::endl;
			point[i] += simulator(AI[i]);
		}
	}

	void mutate() {
		double sum_point = 0;
		for (int i = 0; i < point.size(); i++) {
			sum_point += point[i];
		}

		double random_number;
		std::vector<double> mapping(1,0);
		for (int i = 0; i < point.size(); i++) {
			mapping.push_back(mapping.back() + probability_func(point[i] / sum_point));
		}
		std::vector<Neural_Network> _AI;
		for (int i = 0; i < population; i++) {
			random_number = double(rand() % 10000) / 10000;
			int pos = std::distance(mapping.begin(), std::lower_bound(mapping.begin(), mapping.end(), random_number));
			if (pos > AI.size())
				pos = AI.size();
			else if (pos < 1)
				pos = 1;
			_AI.push_back(AI[pos - 1]);
			_AI.back().mutate(mutation_chance,mutation_rate);
			std::cout << "select with score : " << point[pos - 1] << "\n";
		}
		AI.clear();
		AI = std::move(_AI);

		++generation_number;
	}

	double get_max_point() {
		double get_max = point[0];
		for (int i = 1; i < point.size(); i++) {
			get_max = std::max(point[i], get_max);
		}
		return get_max;
	}

	Neural_Network get_max_model() {
		double get_max = point[0];
		int pos = 0;
		for (int i = 1; i < point.size(); i++) {
			if (point[i] > get_max) {
				pos = i;
				get_max = point[i];
			}
		}
		return AI[pos];
	}

	void print_point() {
		std::vector<double> copy_point = point;
		std::sort(copy_point.begin(), copy_point.end());
		std::cout << "genetic_algortithm : \n";
		for (int i = 0; i < copy_point.size(); i++) {
			std::cout << "top AI : " << i << " got score : " << copy_point[i] << std::endl;
		}
	}



	void set_point(const double& number) {
		for (int i = 0; i < point.size(); i++) {
			point[i] = number;
		}
	}


	void set_population(const double& _population) {
		population = _population;
	}

	void set_mutation_rate(const double& _mutation_rate) {
		mutation_rate = _mutation_rate;
	}

	void set_mutation_chance(const double& _mutation_chance) {
		mutation_chance = _mutation_chance;
	}

	void set_simulator(const std::function<double(const Neural_Network&)>& _simulator) {
		simulator = _simulator;
	}

	void set_probability_func(const std::function<double(double)>& _probability_func) {
		probability_func = _probability_func;
	}

	std::size_t get_generetion_number() {
		return generation_number;
	}

private:
	std::vector<Neural_Network> AI;
	std::vector<double> point;

	std::size_t population = 0;
	double mutation_rate = 0.1;
	double mutation_chance = 0.2;

	std::function<double(Neural_Network&)> simulator;
	std::function<double(double)> probability_func;

	std::size_t generation_number = 0;
};