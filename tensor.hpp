#include <value.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

class Tensor {

  public:

    Tensor(std::vector<unsigned int> dims = std::vector<unsigned int>{1, 1}, std::vector<float>* data = nullptr);

    Tensor* transpose();
    Tensor* _zerograd();    
    void backward();
    Tensor* relu();
    Value* sum();

    static Tensor* matmul(const Tensor &a, const Tensor &b);

    static Tensor zeros(std::vector<unsigned int> dims);
    static Tensor ones(std::vector<unsigned int> dims);
    static Tensor range(std::vector<unsigned int> dims);
    static Tensor rand(std::vector<unsigned int> dims);
    static Tensor randn(std::vector<unsigned int> dims);

    static Tensor zeros_like(const Tensor &other);
    static Tensor ones_like(const Tensor &other);
    static Tensor range_like(const Tensor &other);
    static Tensor rand_like(const Tensor &other);
    static Tensor randn_like(const Tensor &other);

    Tensor* operator+(const Tensor &other);
    Tensor* operator+(const float &f);
    Tensor* operator-(const Tensor &other);
    Tensor* operator-(const float &f);
    Tensor* operator*(const Tensor &other);
    Tensor* operator*(const float &f);
    Tensor* operator/(const Tensor &other);
    Tensor* operator/(const float &f);

    Value* operator[](unsigned int index);
    
    mutable std::vector<Value*> ddata;
    std::vector<unsigned int> dims;
    unsigned int sz;

  private:    

    static void checkInBounds(const Tensor &a, std::vector<unsigned int> point);
    static void checkEqualSize(const Tensor &a, const Tensor &b);
    static void checkMatMulPossible(const Tensor &a, const Tensor &b);
    static void checkTransposePossible(const Tensor &a);
    static void checkBroadcastPossible(const Tensor& a, const Tensor& b);
}; 

Tensor::Tensor(std::vector<unsigned int> dimensions, std::vector<float>* data) {
  
  int size = 1;

  for (int i = 0; i < dimensions.size(); i++) {

    size *= dimensions[i];
  }

  ddata.resize(size);
  if (data) {

    for (int i = 0; i < size; i++) {

      ddata[i] = new Value(&data->operator[](i));
    }
  }

  dims = dimensions;
  sz = size;
}

void Tensor::checkInBounds(const Tensor &a, std::vector<unsigned int> point) {

  if (point.size() != a.dims.size()) {

    std::cout << "Error: Point is not of the correct dimensionality." << std::endl;
    throw 0;
  }

  for (int i = 0; i < a.dims.size(); i++) {

    if (point[i] > a.dims[i]) {

      std::cout << "Error: Index at position " << i << " with value " << point[i] << 
      " is out of bounds." << std::endl;
      throw 0;
    }
  }
}

void Tensor::checkEqualSize(const Tensor &a, const Tensor &b) {

  if (a.dims.size() != b.dims.size()) {

    std::cout << "Error: Tensors must have same dimensionality" << std::endl;
    throw 0;
  }

  for (int i = 0; i < a.dims.size(); i++) {

    if (a.dims[i] != b.dims[i]) {

      std::cout << "Error: Size mismatch at dimension " << i << std::endl;
      throw 0;
    }
  }
}

void Tensor::checkMatMulPossible(const Tensor &a, const Tensor &b) {

  if (a.dims.size() < 2) {
    
    if (b.dims.size() < 2) {

      if (a.dims[0] != b.dims[0]) {

        std::cout << "Error: Size of 1d tensors must match" << std::endl;
        throw 0;
      }
    }

    else {
      
      if (a.dims[0] != b.dims[b.dims.size() -2]) {

        std::cout << "Error: Size of 1d tensor must match other at dimension -2" << std::endl;
        throw 0;
      }
    }
  }

  else if (a.dims.size() == 2) {

    if (b.dims.size() < 2) {

      if (a.dims[1] != b.dims[0]) {

        std::cout << "Error: Size of 1d tensor must match other at dimension -1" << std::endl;
        throw 0;
      }
    }

    else {

      if (a.dims[1] != b.dims[b.dims.size()-2]) {
        
        std::cout << "Error: Size of 2d tensor A at dimension -1 must match size of 2d tensor B at dimension -2" << std::endl;
        throw 0;
      }
    }
  }

  else {

    if (b.dims.size() < 2) {
      
      if (a.dims[a.dims.size()-1] != b.dims[0]) {

        std::cout << "Error: Size of 1d tensor must match other at dimension -1" << std::endl;
        throw 0;
      }
    }

    else if (b.dims.size() == 2) {
      
      if (a.dims[a.dims.size() -1] != b.dims[0]) {

        std::cout << "Error: Size of Nd tensor A at dimension -1 must match size of 2d Tensor B at dimension -2" << std::endl;
        throw 0;
      }
    }

    else {

      if (a.dims[a.dims.size() -1] != b.dims[b.dims.size() -2]) {

        std::cout << "Error: Size of Nd tensor A at dimension -1 must match size of Nd Tensor B at dimension -2" << std::endl;
        throw 0;
      }

      else {
        
        if (a.dims.size() > b.dims.size()) {
          
          for (int i = 3; i < b.dims.size()+1; i++) {

            if (a.dims[a.dims.size() -i] != 1 && b.dims[b.dims.size() -i] != 1) {

              std::cout << "Error: Both tensor have size > 1 at dimension " << i << std::endl;
              throw 0;
            }
          }
        }

        else {

          for (int i = 3; i < a.dims.size()+1; i++) {

            if (a.dims[a.dims.size() -i] != 1 && b.dims[b.dims.size() -i] != 1) {

              std::cout << "Error: Both tensors have size > 1 at dimension " << i << std::endl;
              throw 0;
            }
          }
        }
      }
    }
  }
}

void Tensor::checkTransposePossible(const Tensor &a) {

  if (a.dims.size() != 2) {
    
    std::cout << "Error: Transpose of tensor with rank > 2 is not well defined." << std::endl;
    throw 0;
  }
}

void Tensor::checkBroadcastPossible(const Tensor& a, const Tensor& b) { //checking only for element-wise arithmetic operations 

  unsigned int batch_dims = true;

  for (int ms = 1; ms < std::min(a.dims.size(), b.dims.size()); ms++) {

    if (a.dims[a.dims.size() - ms] == b.dims[b.dims.size() - ms] && batch_dims) {

      continue;
    }
    else {

      if (a.dims[a.dims.size() - ms] != 1 && b.dims[b.dims.size() - ms] != 1) {

        std::cout << "Error: " << std::endl;
        throw 0;
      }

      batch_dims = false;
    }
  }
}

Tensor* Tensor::transpose() {

  checkTransposePossible(*this);
  
  std::vector<unsigned int> swapdims{dims[1], dims[0]};
  Tensor* transposed = new Tensor(swapdims);

  for (int i = 0; i < sz; i++) {

    int j = (i % dims[1])*dims[0] + i/dims[1];
    transposed->ddata[j] = ddata[i]->copy();
  }

  return transposed;
}

Tensor* Tensor::matmul(const Tensor &a, const Tensor &b) {

  checkMatMulPossible(a, b);

  std::vector<unsigned int> adims = a.dims;
  std::vector<unsigned int> bdims = b.dims;

  if (adims.size() == 1) {

    adims.insert(adims.begin(), 1);
  }
  if (bdims.size() == 1) {

    bdims.push_back(1);
  }

  while (bdims.size() != adims.size()) {

    if (adims.size() < bdims.size()) {

      adims.insert(adims.begin(), 1);
    }
    else {

      bdims.insert(bdims.begin(), 1);
    }
  }

  std::vector<unsigned int> resultdims;

  resultdims.insert(resultdims.begin(), bdims[bdims.size() - 1]);
  resultdims.insert(resultdims.begin(), adims[adims.size() - 2]);

  for (int i = 3; i < adims.size() + 1; i++) {

    resultdims.insert(resultdims.begin(), adims[adims.size() - i] * bdims[bdims.size() - i]);
  }

  Tensor* c = new Tensor(resultdims);

  std::function<void(unsigned int&, std::vector<unsigned int>&)> matmul = [&matmul, &a, &b, &c, &resultdims, &adims, &bdims](unsigned int &i, std::vector<unsigned int> &point) -> void {

    if (i < resultdims.size() - 2) {

      for (int j = 0; j < resultdims[i]; j++) {

        std::vector<unsigned int> tpoint = point;
        tpoint.push_back(j);
        unsigned int k = i + 1;
        matmul(k, tpoint);
      }
    }

    else {

      int base = resultdims[i] * resultdims[i + 1];
      int abase = adims[i] * adims[i + 1];
      int bbase = bdims[i] * bdims[i + 1];

      int offset = 0;
      int aoffset = 0;
      int boffset = 0;

      for (int k = point.size()-1; k >= 0; k--) {

        offset += base * point[k];

        if (adims[k] != 1) {

          aoffset += abase * point[k];
        } 
        
        else {
          
          boffset += bbase * point[k];
        }

        base *= resultdims[k];
        abase *= adims[k];
        bbase *= bdims[k];
      }

      unsigned int N = adims[adims.size() -2];
      unsigned int K = adims[adims.size() -1];
      unsigned int M = bdims[bdims.size() -1];

      #pragma omp parallel for
      for (int i = 0; i < N; i++) {

        for (int j = 0; j < M; j++) {
          
          Value* nv = *a.ddata[aoffset + i * K] * *b.ddata[boffset + j];
          for (int k = 1; k < K; k++) {

            nv = nv->operator+=(*(*a.ddata[aoffset + i * K + k] * *b.ddata[boffset + k * M + j]));
          }
          c->ddata[offset + i * resultdims[resultdims.size()-1] + j] = nv;
        }
      }
    }
  };

  std::vector<unsigned int> v;
  unsigned int i = 0;
  matmul(i, v);

  return c;
}

Tensor Tensor::zeros(std::vector<unsigned int> dims) {
    
  Tensor tensor = Tensor(dims);
  const float zero = 0; // ?
  for (int i = 0; i < tensor.sz; i++) {

    tensor.ddata[i] = new Value(&zero);
  }

  return tensor;
}

Tensor Tensor::ones(std::vector<unsigned int> dims) {

  Tensor tensor = Tensor(dims);
  const float one = 1; // ?
  for (int i = 0; i < tensor.sz; i++) {

    tensor.ddata[i] = new Value(&one);
  }

  return tensor;
}

Tensor Tensor::range(std::vector<unsigned int> dims) {

  Tensor tensor = Tensor(dims);
  for (float i = 0; i < tensor.sz; i++) {

    tensor.ddata[i] = new Value(&i);
  }

  return tensor;
}

Tensor Tensor::rand(std::vector<unsigned int> dims) {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1);

  Tensor tensor = Tensor(dims);
  for (int i = 0; i < tensor.sz; i++) {

    float rv = dis(gen);
    tensor.ddata[i] = new Value(&rv);
  }

  return tensor;
}

Tensor Tensor::randn(std::vector<unsigned int> dims) {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> dis(0, 1);

    
  Tensor tensor = Tensor(dims);
  for (int i = 0; i < tensor.sz; i++) {

    float rv = dis(gen);
    tensor.ddata[i] = new Value(&rv);
  }

  return tensor;
}

Tensor Tensor::zeros_like(const Tensor &other) {

  return Tensor::zeros(other.dims);
}

Tensor Tensor::ones_like(const Tensor &other) {

  return Tensor::ones(other.dims);
}

Tensor Tensor::range_like(const Tensor &other) {

  return Tensor::range(other.dims);
}

Tensor Tensor::rand_like(const Tensor &other) {

  return Tensor::rand(other.dims);
}  

Tensor Tensor::randn_like(const Tensor &other) {

  return Tensor::randn(other.dims);
}  

void Tensor::backward() {
  
#pragma omp parallel for
  for (int i = 0; i < sz; i++) {
    ddata[i]->backward();
  }
}

Tensor* Tensor::relu() {

  Tensor* result = new Tensor(dims);
  for (int i = 0; i < sz; i++) {

    result->ddata[i] = ddata[i]->relu();
  }

  return result;
}

Tensor* Tensor::_zerograd() {

  std::vector<float> v;
  v.resize(sz);
  for (int i = 0; i < sz; i++) {

    v[i] = ddata[i]->val;
  }

  return new Tensor(dims, &v);
}

Value* Tensor::sum() {
  
  float zero = 0;
  Value* vp = new Value(&zero);
  for (int i = 0; i < sz; i++) {
    
    vp = *vp + *ddata[i];
  }

  return vp;
}

Value* Tensor::operator[](unsigned int i) {

  return ddata[i];
}

Tensor* Tensor::operator+(const Tensor &other) {

  checkBroadcastPossible(*this, other);

  std::vector<unsigned int> adims = this->dims;
  std::vector<unsigned int> bdims = other.dims;


  while (adims.size() != bdims.size()) {

    if (adims.size() < bdims.size()) {

      adims.insert(adims.begin(), 1);
    }
    else {

      bdims.insert(bdims.begin(), 1);
    }
  }

  std::vector<unsigned int> resultdims;

  for (int i = 0; i < adims.size(); i++) {

    resultdims.insert(resultdims.end(), std::max(adims[i], bdims[i]));
  }

  Tensor* c = new Tensor(resultdims);

  std::function<void(unsigned int&, std::vector<unsigned int>&)> operate = [&operate, this, &other, &c, &resultdims, &adims, &bdims](unsigned int& i, std::vector<unsigned int>& point) -> void {

    if (i < resultdims.size() - std::min(this->dims.size(), other.dims.size())) {

      for (int j = 0; j < resultdims[i]; j++) {

        std::vector<unsigned int> tpoint = point;
        tpoint.push_back(j);
        unsigned int k = i + 1;
        operate(k, tpoint);
      }
    }
    else {

      int base = 1;
      int abase = 1;
      int bbase = 1;

      while (i < adims.size()) {

        base *= resultdims[i];
        abase *= adims[i];
        bbase *= bdims[i];
        i++;
      }

      int offset = 0;
      int aoffset = 0;
      int boffset = 0;

      for (int k = point.size() - 1; k >= 0; k--) {

        offset += base * point[k];

        if (adims[k] != 1) {

          aoffset += abase * point[k];
        }
        else {

          boffset += bbase * point[k];
        }

        base *= resultdims[k];
        abase *= adims[k];
        bbase *= bdims[k];
      }

      for (int i = 0; i < std::min(this->sz, other.sz); i++) {

        c->ddata[offset + i] = *this->ddata[aoffset + i] + *other.ddata[boffset + i];
      }
    }
  };

  std::vector<unsigned int> v;
  unsigned int i = 0;
  operate(i, v);

  return c;
}

Tensor* Tensor::operator+(const float &f) {

  Tensor* other = new Tensor(dims);
  for (int i = 0; i < sz; i++) {

    Value* nv = new Value(&f);
    other->ddata[i] = nv;
  }

  return *this + *other;
}

Tensor* Tensor::operator-(const Tensor &other) {

  checkBroadcastPossible(*this, other);

  std::vector<unsigned int> adims = this->dims;
  std::vector<unsigned int> bdims = other.dims;


  while (adims.size() != bdims.size()) {

    if (adims.size() < bdims.size()) {

      adims.insert(adims.begin(), 1);
    }
    else {

      bdims.insert(bdims.begin(), 1);
    }
  }

  std::vector<unsigned int> resultdims;

  for (int i = 0; i < adims.size(); i++) {

    resultdims.insert(resultdims.end(), std::max(adims[i], bdims[i]));
  }

  Tensor* c = new Tensor(resultdims);

  std::function<void(unsigned int&, std::vector<unsigned int>&)> operate = [&operate, this, &other, &c, &resultdims, &adims, &bdims](unsigned int& i, std::vector<unsigned int>& point) -> void {

    if (i < resultdims.size() - std::min(this->dims.size(), other.dims.size())) {

      for (int j = 0; j < resultdims[i]; j++) {

        std::vector<unsigned int> tpoint = point;
        tpoint.push_back(j);
        unsigned int k = i + 1;
        operate(k, tpoint);
      }
    }
    else {

      int base = 1;
      int abase = 1;
      int bbase = 1;

      while (i < adims.size()) {

        base *= resultdims[i];
        abase *= adims[i];
        bbase *= bdims[i];
        i++;
      }

      int offset = 0;
      int aoffset = 0;
      int boffset = 0;

      for (int k = point.size() - 1; k >= 0; k--) {

        offset += base * point[k];

        if (adims[k] != 1) {

          aoffset += abase * point[k];
        }
        else {

          boffset += bbase * point[k];
        }

        base *= resultdims[k];
        abase *= adims[k];
        bbase *= bdims[k];
      }

      for (int i = 0; i < std::min(this->sz, other.sz); i++) {

        c->ddata[offset + i] = *this->ddata[aoffset + i] - *other.ddata[boffset + i];
      }
    }
  };

  std::vector<unsigned int> v;
  unsigned int i = 0;
  operate(i, v);

  return c;
}

Tensor* Tensor::operator-(const float &f) {

  Tensor* other = new Tensor(dims);
  for (int i = 0; i < sz; i++) {

    Value* nv = new Value(&f);
    other->ddata[i] = nv;
  }

  return *this - *other;
}


Tensor* Tensor::operator*(const Tensor &other) {

  checkBroadcastPossible(*this, other);

  std::vector<unsigned int> adims = this->dims;
  std::vector<unsigned int> bdims = other.dims;


  while (adims.size() != bdims.size()) {

    if (adims.size() < bdims.size()) {

      adims.insert(adims.begin(), 1);
    }
    else {

      bdims.insert(bdims.begin(), 1);
    }
  }

  std::vector<unsigned int> resultdims;

  for (int i = 0; i < adims.size(); i++) {

    resultdims.insert(resultdims.end(), std::max(adims[i], bdims[i]));
  }

  Tensor* c = new Tensor(resultdims);

  std::function<void(unsigned int&, std::vector<unsigned int>&)> operate = [&operate, this, &other, &c, &resultdims, &adims, &bdims](unsigned int& i, std::vector<unsigned int>& point) -> void {

    if (i < resultdims.size() - std::min(this->dims.size(), other.dims.size())) {

      for (int j = 0; j < resultdims[i]; j++) {

        std::vector<unsigned int> tpoint = point;
        tpoint.push_back(j);
        unsigned int k = i + 1;
        operate(k, tpoint);
      }
    }
    else {

      int base = 1;
      int abase = 1;
      int bbase = 1;

      while (i < adims.size()) {

        base *= resultdims[i];
        abase *= adims[i];
        bbase *= bdims[i];
        i++;
      }

      int offset = 0;
      int aoffset = 0;
      int boffset = 0;

      for (int k = point.size() - 1; k >= 0; k--) {

        offset += base * point[k];

        if (adims[k] != 1) {

          aoffset += abase * point[k];
        }
        else {

          boffset += bbase * point[k];
        }

        base *= resultdims[k];
        abase *= adims[k];
        bbase *= bdims[k];
      }

      for (int i = 0; i < std::min(this->sz, other.sz); i++) {

        c->ddata[offset + i] = *this->ddata[aoffset + i] * *other.ddata[boffset + i];
      }
    }
  };

  std::vector<unsigned int> v;
  unsigned int i = 0;
  operate(i, v);

  return c;
}

Tensor* Tensor::operator*(const float &f) {

  Tensor* other = new Tensor(dims);
  for (int i = 0; i < sz; i++) {

    Value* nv = new Value(&f);
    other->ddata[i] = nv;
  }
  
  return *this * *other;
}

Tensor* Tensor::operator/(const Tensor &other) {

  checkBroadcastPossible(*this, other);

  std::vector<unsigned int> adims = this->dims;
  std::vector<unsigned int> bdims = other.dims;


  while (adims.size() != bdims.size()) {

    if (adims.size() < bdims.size()) {

      adims.insert(adims.begin(), 1);
    }
    else {

      bdims.insert(bdims.begin(), 1);
    }
  }

  std::vector<unsigned int> resultdims;

  for (int i = 0; i < adims.size(); i++) {

    resultdims.insert(resultdims.end(), std::max(adims[i], bdims[i]));
  }

  Tensor* c = new Tensor(resultdims);

  std::function<void(unsigned int&, std::vector<unsigned int>&)> operate = [&operate, this, &other, &c, &resultdims, &adims, &bdims](unsigned int& i, std::vector<unsigned int>& point) -> void {

    if (i < resultdims.size() - std::min(this->dims.size(), other.dims.size())) {

      for (int j = 0; j < resultdims[i]; j++) {

        std::vector<unsigned int> tpoint = point;
        tpoint.push_back(j);
        unsigned int k = i + 1;
        operate(k, tpoint);
      }
    }
    else {

      int base = 1;
      int abase = 1;
      int bbase = 1;

      while (i < adims.size()) {

        base *= resultdims[i];
        abase *= adims[i];
        bbase *= bdims[i];
        i++;
      }

      int offset = 0;
      int aoffset = 0;
      int boffset = 0;

      for (int k = point.size() - 1; k >= 0; k--) {

        offset += base * point[k];

        if (adims[k] != 1) {

          aoffset += abase * point[k];
        }
        else {

          boffset += bbase * point[k];
        }

        base *= resultdims[k];
        abase *= adims[k];
        bbase *= bdims[k];
      }

      for (int i = 0; i < std::min(this->sz, other.sz); i++) {

        c->ddata[offset + i] = *this->ddata[aoffset + i] / *other.ddata[boffset + i];
      }
    }
  };

  std::vector<unsigned int> v;
  unsigned int i = 0;
  operate(i, v);

  return c;
}
    
Tensor* Tensor::operator/(const float &f) {

  Tensor* other = new Tensor(dims);
  for (int i = 0; i < sz; i++) {

    Value* nv = new Value(&f);
    other->ddata[i] = nv;
  }

  return *this / *other;
}