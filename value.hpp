#include <cmath>
#include <stack>
#include <unordered_set>
#include <functional>

enum Operation {addition, multiplication, exponentiation, ReLU, Variable};

class Value {
  
  public:
  
    Value(const float* data, const Value* parent1 = nullptr, const Value* parent2 = nullptr, Operation operation = Operation::Variable);

    Value* relu() const;
    Value* copy() const;
    void backward() const;
    void _backward() const;

    const bool operator==(const Value &other) const;
    const bool operator!=(const Value &other) const;
    Value operator=(const Value &other) const;
    Value* operator+=(const Value& other) const;
    Value* operator+(const Value &other) const;
    Value* operator+(const float &f) const;
    Value* operator-(const Value &other) const;
    Value* operator-(const float &f) const;
    Value* operator*(const Value &other) const;
    Value* operator*(const float &f) const;
    Value* operator/(const Value &other) const;
    Value* operator/(const float &f) const;
    Value* pow(const float &f) const;

    float val;
    mutable float grad;
    const Value* p1;
    const Value* p2;

  private:

    Operation op;
};

Value::Value(const float* data, const Value *parent1, const Value *parent2, Operation operation) {

  float* nvalue = new float();

  if(data){
    std::memcpy(nvalue, data, sizeof(float));
  }

  val = *nvalue;
  op = operation;
  p1 = parent1;
  p2 = parent2;
  grad = 0;
}

void Value::backward() const {

  grad = 1;

  struct hasher {

    size_t operator()(const Value& value) const {

      return std::hash<float>()(value.val);
    }
  };

  std::unordered_set<Value, hasher> seen;
  std::stack<const Value*> backward;

  std::function<void(const Value&)> walk = [&walk, &seen, &backward](const Value &v) -> void {

    if (seen.find(v) == seen.end()) {

      seen.insert(v);
      if (v.op != Operation::Variable) {

        walk(*v.p1);
        walk(*v.p2);
      }

      backward.push(&v);
    }
  };

  walk(*this);

  while (!backward.empty()) {
    
    const Value& v = *backward.top();
    backward.pop();
    v._backward();
  }
}

void Value::_backward() const  {

  switch(op) {

    case Operation::addition: {

      p1->grad += grad;
      p2->grad += grad;
    } break;

    case Operation::multiplication: {

      p1->grad += p2->val * grad;
      p2->grad += p1->val * grad;
    } break;

    case Operation::exponentiation: {
      
      p1->grad += (p2->val * std::pow(p1->val, (p2->val - 1))) * grad;
    } break;

    case Operation::ReLU: {
      
      if (val > 0) {

        p1->grad += grad;
      }
    } break;

    case Operation::Variable: {
      
      return;
    } break;
  }
}

Value* Value::relu() const {

  float nv;

  if (this->val > 0) {

    nv = this->val;
  }
  else {

    nv = 0;
  }

  Value* result = new Value(&nv, this, this, Operation::ReLU);
  return result;
}

Value* Value::copy() const {

  return new Value(&val, p1, p2, op);
}


const bool Value::operator!=(const Value &other) const {

  return this != &other;
}

const bool Value::operator==(const Value &other) const {
  
  return this == &other;
}

Value Value::operator=(const Value &other) const {

  return other;
}

Value* Value::operator+=(const Value& other) const {

  return *this + other;
}

Value* Value::operator+(const Value &other) const {

  float nv = this->val + other.val;
  Value* result = new Value(&nv, this, &other, Operation::addition);
  return result;
}

Value* Value::operator+(const float &f) const {

  static const Value other = Value(&f, &other, &other, Operation::Variable);

  float nv = this->val + other.val;
  Value* result = new Value(&nv, this, &other, Operation::addition);
  return result;
}

Value* Value::operator-(const Value &other) const {
  
  Value* nv = other * -1;
  Value* result = *this + *nv;
  return result;
}

Value* Value::operator-(const float &f) const {

  Value* other = new Value(&f);
  
  Value* nv = *other * -1;
  Value* result = *this + *nv;
  return result;
}

Value* Value::operator*(const Value &other) const {

  float nv = this->val * other.val;
  Value* result = new Value(&nv, this, &other, Operation::multiplication);
  return result;
}

Value* Value::operator*(const float &f) const {
  
  Value* other = new Value(&f);

  float nv = this->val * other->val;
  Value* result = new Value(&nv, this, other, Operation::multiplication);
  return result;
}

Value* Value::operator/(const Value &other) const {
  
  Value* nv = other.pow(-1);
  Value* result = *this * *nv;
  return result;
}

Value* Value::operator/(const float &f) const {
  
  Value* other = new Value(&f);
  
  Value* nv = other->pow(-1);
  Value* result = *this * *nv;
  return result;
}

Value* Value::pow(const float &f) const {

  Value* other = new Value(&f);

  float nv = std::pow(this->val, f);
  Value* result = new Value(&nv, this, other, Operation::exponentiation);
  return result;
}