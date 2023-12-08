#include <value.hpp>
#include <vector>
#include <random>

class Tensor {

  public:

    Tensor(unsigned int rows = 1, unsigned int cols = 1, float* data = nullptr);

    Tensor* transpose();
    Tensor* relu();
    void backward();
    void _insert(Value &v, int i);

    static Tensor* matmul(const Tensor &A, const Tensor &B);

    static Tensor zeros(unsigned int rows, unsigned int cols);
    static Tensor ones(unsigned int rows, unsigned int cols);
    static Tensor range(unsigned int rows, unsigned int cols);
    static Tensor rand(unsigned int rows, unsigned int cols);
    static Tensor randn(unsigned int rows, unsigned int cols);

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
    Value* operator()(unsigned int x, unsigned int y);
    
    mutable std::vector<Value*> ddata;
    unsigned int r;
    unsigned int c;
    std::vector<float> grad;

  private:    

    static void checkEqualSize(const Tensor &a, const Tensor &b);
    static void checkMatMulPossible(const Tensor &a, const Tensor &b);
    static void checkInBounds(const Tensor &a, unsigned int x, unsigned int y);
}; 

Tensor::Tensor(unsigned int rows, unsigned int cols, float* data) {
    
  ddata.resize(rows*cols);
  if (data) {

    for (int i = 0; i < rows*cols; i++) {

      ddata[i] = new Value(&data[i]);
    }
  }

  r = rows;
  c = cols;
}

void Tensor::checkInBounds(const Tensor &a, unsigned int x, unsigned int y) {

  if (x >= a.r || y >= a.c) {

    std::cout << "Error: Index out of bounds."<< std::endl;
    throw 0;
  }
}

void Tensor::checkEqualSize(const Tensor &a, const Tensor &b) {

  if (a.r != b.r || a.c != b.c) {

    std::cout << "Error: Tensors must have same dimensionality" << std::endl;
    throw 0;
  }
}

void Tensor::checkMatMulPossible(const Tensor &a, const Tensor &b) {

  if (a.c != b.r) {

    std::cout << "Error: A.cols() = " << a.c << " != B.rows() = "<< b.r << std::endl;
    throw 0;
  }
}

Tensor* Tensor::transpose() {

  Tensor* transposed = new Tensor(c, r);
  for (int i = 0; i < c * r; i++) {

    int j = (i % c)*r + i/c;
    transposed->ddata[j] = ddata[i]->copy();
  }

  return transposed;
}

Tensor* Tensor::matmul(const Tensor &a, const Tensor &b) {

  checkMatMulPossible(a, b);
  unsigned int n = a.r;
  unsigned int k = a.c;
  unsigned int m = b.c;

  Tensor* c = new Tensor(n, m);
  #pragma omp parallel for
      for (int i = 0; i < n; i++) {

        for (int j = 0; j < m; j++) {
      
          Value* nv = *a.ddata[i * a.c] * *b.ddata[j];
          for (int k = 1; k < k; k++) {

            nv = nv->operator+=(*(*a.ddata[i * a.c + k] * *b.ddata[k * b.c + j]));
          }
          c->ddata[i * c->c + j] = nv;
        }
      }

  return c;
}

Tensor Tensor::zeros(unsigned int rows, unsigned int cols) {
    
  Tensor tensor = Tensor(rows, cols);
  const float zero = 0; // ?
  for (int i = 0; i < rows*cols; i++) {

    tensor.ddata[i] = new Value(&zero);
  }

  return tensor;
}

Tensor Tensor::ones(unsigned int rows, unsigned int cols) {

  Tensor tensor = Tensor(rows, cols);
  const float one = 1; // ?
  for (int i = 0; i < cols*rows; i++) {

    tensor.ddata[i] = new Value(&one);
  }

  return tensor;
}

Tensor Tensor::range(unsigned int rows, unsigned int cols) {

  Tensor tensor = Tensor(rows,cols);
  for (float i = 0; i < cols*rows; i++) {

    tensor.ddata[i] = new Value(&i);
  }

  return tensor;
}

Tensor Tensor::rand(unsigned int rows, unsigned int cols) {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1);

  Tensor tensor = Tensor(rows,cols);
  for (int i = 0; i < cols*rows; i++) {

    float rv = dis(gen);
    tensor.ddata[i] = new Value(&rv);
  }

  return tensor;
}

Tensor Tensor::randn(unsigned int rows, unsigned int cols){

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> dis(0, 1);

    
  Tensor tensor = Tensor(rows,cols);
  for (int i = 0; i < cols*rows; i++) {

    float rv = dis(gen);
    tensor.ddata[i] = new Value(&rv);
  }

  return tensor;
}

Tensor Tensor::zeros_like(const Tensor &other){

  return Tensor::zeros(other.r, other.c);
}

Tensor Tensor::ones_like(const Tensor &other){

  return Tensor::ones(other.r, other.c);
}

Tensor Tensor::range_like(const Tensor &other){

  return Tensor::range(other.r, other.c);
}

Tensor Tensor::rand_like(const Tensor &other){

  return Tensor::rand(other.r, other.c);
}  

Tensor Tensor::randn_like(const Tensor &other){

  return Tensor::randn(other.r, other.c);
}  

void Tensor::backward() {
  
  grad.resize(r*c);
  #pragma omp parallel for
    for (int i = 0; i < r*c; i++) {
      
      ddata[i]->backward();
      grad[i] = ddata[i]->grad;
    }
}

Tensor* Tensor::relu() {

  Tensor* result = new Tensor(r, c);
  for (int i = 0; i < r*c; i++) {

    result->ddata[i] = ddata[i]->relu();
  }

  return result;
}

void Tensor::_insert(Value &v, int i) {
  
  ddata[i] = &v;
}

Value* Tensor::operator[](unsigned int i) {

  return ddata[i];
}

Value* Tensor::operator()(unsigned int x, unsigned int y) {

  checkInBounds(*this,x,y);
  return ddata[x*c + y];
}

Tensor* Tensor::operator+(const Tensor &other) {

  checkEqualSize(*this,other);
  Tensor* result = new Tensor(r, c);
  for (int i = 0; i < r*c; i++) {
    
    result->ddata[i] = *this->ddata[i] + *other.ddata[i];
    }

  return result;
}

Tensor* Tensor::operator+(const float &f) {
  
  Tensor* other = new Tensor(r, c);
  for (int i = 0; i < r*c; i++) {

    Value* nv = new Value(&f);
    other->ddata[i] = nv;
    }

  Tensor* result = new Tensor(r, c);
  for (int i = 0; i < r*c; i++) {
    
    result->ddata[i] = *this->ddata[i] + *other->ddata[i];
    }

  return result;
}

Tensor* Tensor::operator-(const Tensor &other) {

  checkEqualSize(*this,other);
  Tensor* result = new Tensor(r, c);
  for (int i = 0; i < r*c; i++) {
    
    result->ddata[i] = *this->ddata[i] - *other.ddata[i];
    }

  return result;
}

Tensor* Tensor::operator-(const float &f) {

  Tensor* other = new Tensor(r, c);
  for (int i = 0; i < r*c; i++) {

    Value* nv = new Value(&f);
    other->ddata[i] = nv;
    }

  Tensor* result = new Tensor(r, c);
  for (int i = 0; i < r*c; i++) {
    
    result->ddata[i] = *this->ddata[i] - *other->ddata[i];
    }

  return result;
}


Tensor* Tensor::operator*(const Tensor &other) {

  checkEqualSize(*this,other);
  Tensor* result = new Tensor(r, c);
  for (int i = 0; i < r*c; i++) {

    result->ddata[i] = *this->ddata[i] * *other.ddata[i];
    }

  return result;
}

Tensor* Tensor::operator*(const float &f) {

  Tensor* other = new Tensor(r, c);
  for (int i = 0; i < r*c; i++) {

    Value* nv = new Value(&f);
    other->ddata[i] = nv;
    }

  Tensor* result = new Tensor(r, c);
  for (int i = 0; i < r*c; i++) {
    
    result->ddata[i] = *this->ddata[i] - *other->ddata[i];
    }

  return result;
}

Tensor* Tensor::operator/(const Tensor &other) {

  checkEqualSize(*this,other);
  Tensor* result = new Tensor(r, c);
  for (int i = 0; i < r*c; i++) {
    
    result->ddata[i] = *this->ddata[i] / *other.ddata[i];
    }

  return result;
}
    
Tensor* Tensor::operator/(const float &f) {

  Tensor* other = new Tensor(r, c);
  for (int i = 0; i < r*c; i++) {

    Value* nv = new Value(&f);
    other->ddata[i] = nv;
    }

  Tensor* result = new Tensor(r, c);
  for (int i = 0; i < r*c; i++) {
    
    result->ddata[i] = *this->ddata[i] / *other->ddata[i];
    }
    
  return result;
}