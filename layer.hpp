#include <tensor.hpp>

enum Linearity {Linear, Nonlinear};

class Layer {

public:

  Layer(unsigned int din, unsigned int dout, Linearity linearity);
  Tensor* operator()(Tensor& x);
  Tensor* operator[](unsigned int i);

  void _backward(float lr);

  std::vector<Tensor> params;

private:

  Linearity lin;
  unsigned int outd;
};

Layer::Layer(unsigned int din, unsigned int dout, Linearity linearity) {

  params.resize(2);
  std::vector<unsigned int>  adims{din, dout};
  std::vector<unsigned int> bdims{30, 1, dout}; // no broadcasting implementation for element-wise arithmetic operations,
  params[0] = Tensor::rand(adims); //              batch dimension is hardcoded here for purposes of example
  params[1] = Tensor::zeros(bdims);

  outd = dout;
  lin = linearity;
}

Tensor* Layer::operator()(Tensor& x) {

  Tensor* t = Tensor::matmul(x, params[0]);
  Tensor* a = t->operator+(params[1]);

  switch (lin) {

    case Linear: {

      return a;
    } break;

    case Nonlinear: {

      Tensor* res = a->relu();
      return res;
    } break;
  }
}

Tensor* Layer::operator[](unsigned int i) {

  return &params[i];
}

void Layer::_backward(float lr = 0.001F) {

  for (int i = 0; i < params.size(); i++) {
    
    for (int j = 0; j < params[i].dims[params[i].dims.size()-2] * params[i].dims[params[i].dims.size()-1]; j++) {

      params[i][j]->val = params[i][j]->val - params[i][j]->grad * lr;
    }

    params[i]._zerograd();
  }
}