# uname01

## Tensor Class with Automatic Differentiation

N-Dimensional tensor with automatic differentiation. Performs flexible batched matrix multiplication using broadcasting with **`Tensor::matmul()`** method. 

Value class constructs computational graph while performing arithmetic operations. Upon **`.backward()`** call traverses graph backwards performing chain rule to calculate gradients.

```c++
  #include "tensor.hpp"
  #include "iostream"
  
  void main() {

	std::vector<unsigned int> dims1{5, 2, 4};
	Tensor a = Tensor::rand(dims1);

	std::vector<unsigned int> dims2{4, 2};
	Tensor b = Tensor::rand(dims2);

	Tensor* c = Tensor::matmul(a, b);
	std::cout << c->dims[0] << "x" << c->dims[1] << "x" << c->dims[2] << "\n" << std::endl;

	Value* r = c->sum();
	r->backward();
	std::cout << a[0]->grad<< std::endl;
  }
```

```bash
  5x2x2

  1.06317
```

### Caveats

Batched matmul is only available for pairs of tensors wherein, for all batch dimensions, at least 1 tensor is of size 1 at that dimension. When called on tensors of different rank, matmul extends lower rank tensor to size of 1 in the approporiate dimensions until ranks match.

### What's next

Einstein summation notation, basic neural network architechtures.