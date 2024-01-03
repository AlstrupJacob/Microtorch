#include <layer.hpp>

class nn {

  public:

    nn(std::vector<int> dims);
    Tensor* operator()(Tensor& x);
    Layer* operator[](unsigned int i);

    void _backward(float lr);
    std::vector<Layer*> layers;

  private:

    int d;
};

nn::nn(std::vector<int> dims) {

  int n = dims.size();
  layers.resize(n - 1);
  for (int i = 0; i < n - 1; i++) {

    if (i != n - 2) {

      layers[i] = new Layer(dims[i], dims[i + 1], Linearity::Nonlinear);
    }

    else {

      layers[i] = new Layer(dims[i], dims[i + 1], Linearity::Linear);
    }
  }

  d = n;
}

Tensor* nn::operator()(Tensor& x) {

  Tensor* tp = &x;
  for (int i = 0; i < d - 1; i++) {

    tp = layers[i]->operator()(*tp);
  }

  return tp;
}

Layer* nn::operator[](unsigned int i) {

  return layers[i];
}

void nn::_backward(float lr = 0.001F) {

  for (int i = 0; i < layers.size(); i++) {

    layers[i]->_backward(lr);
  }
}