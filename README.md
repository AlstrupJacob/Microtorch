# uname01

## Tensor Class with Automatic Differentiation

Constructs a computational graph while performing arithmetic operations. Upon **`.backward()`** call traverses graph backwards performing chain rule to calculate gradients.

```c++
  #include "tensor.hpp"
  #include "iostream"
  
  void main() {
    Tensor a = Tensor::rand(18, 4);
    Tensor* b = a.transpose();
    Tensor* c = Tensor::matmul(a, *b);
    
    c->backward();
    std::cout << a[0]->grad << std::endl;
  }
```

```bash
  8.51811
```