#pragma once

#include "Header.h"
#include "Layer.cpp"

class LayerId {	// this class is used as API to set the layer setting
public:
	LayerId(Layer::type _Layer_type, const std::size_t& _Layer_size, std::string _setting = "") : Layer_type(_Layer_type), Layer_size(_Layer_size) {
		setting = _setting;
	}

	Layer::type Layer_type;
	std::size_t Layer_size;
	std::string setting;
};