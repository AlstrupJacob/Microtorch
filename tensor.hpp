#include <value.hpp>
#include <iostream>
#include <vector>
#include <random>

class Tensor {

  public:

    Tensor(std::vector<unsigned int> dims, std::vector<float>* data = nullptr);

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

  std::vector<unsigned int> atds = a.dims;
  std::vector<unsigned int> btds = b.dims;

  if (atds.size() == 1) {

    atds.insert(atds.begin(), 1);
  }

  if (btds.size() == 1) {
    
    btds.push_back(1);
  }

  while (btds.size() != atds.size()) {

    if (atds.size() < btds.size()) {

      atds.insert(atds.begin(), 1);
    }

    else {
      
      btds.insert(btds.begin(), 1);
    }
  }

  std::vector<unsigned int> rds;  
  
  rds.insert(rds.begin(), btds[btds.size() -1]);
  rds.insert(rds.begin(), atds[atds.size() -2]);
  
  for (int i = 3; i < atds.size() + 1; i++) {

    rds.insert(rds.begin(), atds[atds.size() -i] * btds[btds.size() -i]);
  }

  Tensor* c = new Tensor(rds);

  std::function<void(unsigned int&, std::vector<unsigned int>&)> matmul = [&matmul, &a, &b, &c, &rds, &atds, &btds](unsigned int &i, std::vector<unsigned int> &point) -> void {

    if (i < rds.size() - 2) {

      for (int j = 0; j < rds[i]; j++) {

        std::vector<unsigned int> tpoint = point;
        tpoint.push_back(j);
        unsigned int k = i + 1;
        matmul(k, tpoint);
      }
    }

    else {

      int base = rds[i] * rds[i + 1];
      int abase = atds[i] * atds[i + 1];
      int bbase = btds[i] * btds[i + 1];

      int offset = 0;
      int aoffset = 0;
      int boffset = 0;

      for (int k = point.size()-1; k >= 0; k--) {

        offset += base * point[k];

        if (atds[k] != 1) {

          aoffset += abase * point[k];
        } 
        
        else {
          
          boffset += bbase * point[k];
        }

        base *= rds[k];
        abase *= atds[k];
        bbase *= btds[k];
      }

      unsigned int N = atds[atds.size() -2];
      unsigned int K = atds[atds.size() -1];
      unsigned int M = btds[btds.size() -1];

      #pragma omp parallel for
      for (int i = 0; i < N; i++) {

        for (int j = 0; j < M; j++) {
          
          Value* nv = *a.ddata[aoffset + i * K] * *b.ddata[boffset + j];
          for (int k = 1; k < K; k++) {

            nv = nv->operator+=(*(*a.ddata[aoffset + i * K + k] * *b.ddata[boffset + k * M + j]));
          }
          c->ddata[offset + i * rds[rds.size()-1] + j] = nv;
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

  checkEqualSize(*this, other);
  Tensor* result = new Tensor(dims);
  for (int i = 0; i < sz; i++) {
    
    result->ddata[i] = *this->ddata[i] + *other.ddata[i];
    }

  return result;
}

Tensor* Tensor::operator+(const float &f) {
  
  Tensor* other = new Tensor(dims);
  for (int i = 0; i < sz; i++) {

    Value* nv = new Value(&f);
    other->ddata[i] = nv;
    }

  Tensor* result = new Tensor(dims);
  for (int i = 0; i < sz; i++) {
    
    result->ddata[i] = *this->ddata[i] + *other->ddata[i];
    }

  return result;
}

Tensor* Tensor::operator-(const Tensor &other) {

  checkEqualSize(*this,other);
  Tensor* result = new Tensor(dims);
  for (int i = 0; i < sz; i++) {
    
    result->ddata[i] = *this->ddata[i] - *other.ddata[i];
    }

  return result;
}

Tensor* Tensor::operator-(const float &f) {

  Tensor* other = new Tensor(dims);
  for (int i = 0; i < sz; i++) {

    Value* nv = new Value(&f);
    other->ddata[i] = nv;
    }

  Tensor* result = new Tensor(dims);
  for (int i = 0; i < sz; i++) {
    
    result->ddata[i] = *this->ddata[i] - *other->ddata[i];
    }

  return result;
}


Tensor* Tensor::operator*(const Tensor &other) {

  checkEqualSize(*this,other);
  Tensor* result = new Tensor(dims);
  for (int i = 0; i < sz; i++) {

    result->ddata[i] = *this->ddata[i] * *other.ddata[i];
    }

  return result;
}

Tensor* Tensor::operator*(const float &f) {

  Tensor* other = new Tensor(dims);
  for (int i = 0; i < sz; i++) {

    Value* nv = new Value(&f);
    other->ddata[i] = nv;
    }

  Tensor* result = new Tensor(dims);
  for (int i = 0; i < sz; i++) {
    
    result->ddata[i] = *this->ddata[i] - *other->ddata[i];
    }

  return result;
}

Tensor* Tensor::operator/(const Tensor &other) {

  checkEqualSize(*this,other);
  Tensor* result = new Tensor(dims);
  for (int i = 0; i < sz; i++) {
    
    result->ddata[i] = *this->ddata[i] / *other.ddata[i];
    }

  return result;
}
    
Tensor* Tensor::operator/(const float &f) {

  Tensor* other = new Tensor(dims);
  for (int i = 0; i < sz; i++) {

    Value* nv = new Value(&f);
    other->ddata[i] = nv;
    }

  Tensor* result = new Tensor(dims);
  for (int i = 0; i < sz; i++) {
    
    result->ddata[i] = *this->ddata[i] / *other->ddata[i];
    }
    
  return result;
}