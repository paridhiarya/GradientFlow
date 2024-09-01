from visualization import *
import math
import random

class Value:
  
  def __init__(self, data, _children=(), _operator='', label=''):
    self.data = data
    self.grad = 0.0
    self._backward = lambda: None
    self._children = set(_children)
    self._operator = _operator
    self.label = label

  def __repr__(self):
    return f"Value(data={self.data})"
  
  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')
    
    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward
    
    return out

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')
    
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
      
    return out
  
  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Value(self.data**other, (self,), f'**{other}')

    def _backward():
        self.grad += other * (self.data ** (other - 1)) * out.grad
    out._backward = _backward

    return out
  
  def __radd__(self, other): # other + self
    return self + other
  
  def __rmul__(self, other): # other * self
    return self * other

  def __truediv__(self, other): # self / other
    return self * other**-1

  def __rtruediv__(self, other): # other / self
    return other * self**-1

  def __neg__(self): # -self
    return self * -1

  def __sub__(self, other): # self - other
    return self + (-other)
  
  def __rsub__(self, other): # other - self
    return other + (-self)

  def exp(self):
    x = self.data
    out = Value(math.exp(x), (self, ), 'exp')
    
    def _backward():
      self.grad += out.data * out.grad 
    out._backward = _backward
    
    return out
  
  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Value(t, (self, ), 'tanh')
    
    def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward
    
    return out
  
  def relu(self):
    out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
    
    def _backward():
      self.grad += (out.data > 0) * out.grad
      
    out._backward = _backward
    
    return out
  
  
  def backward(self):
    
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._children:
          build_topo(child)
        topo.append(v)
    build_topo(self)
    
    self.grad = 1.0
    for node in reversed(topo):
      node._backward()
    
class Module:
  def zero_grad(self):
    for p in self.parameters():
      p.grad = 0

  def parameters(self):
    return []

class Neuron(Module):
  def __init__(self, nin, act_func='R',nonlin=True):
    self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1, 1))
    self.nonlin = nonlin
    self.activation_function = act_func
    
  def __call__(self, x):
    act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b) 
    if (self.activation_function == 'R'):
      return act.relu() if self.nonlin else act
    return act.tanh() if self.nonlin else act
  
  def parameters(self):
    return self.w + [self.b]
  
  def __repr__(self):
    return f"{('ReLU' if self.activation_function=='R' else 'Tanh') if self.nonlin else 'Linear'} Neuron({len(self.w)})"

  
class Layer(Module):
  def __init__(self, nin, nout, act_func='R'):
    self.neurons = [Neuron(nin, act_func) for _ in range(nout)]
    
  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs
  
  def parameters(self):
    return [p for n in self.neurons for p in n.parameters()]
  
  def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    
class MLP(Module):
  def __init__(self, nin, nout, act_func='R'):
    sz = [nin] + nout
    self.activation_function = act_func
    self.layers = [Layer(sz[i], sz[i+1], act_func) for i in range(len(nout))]
    
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  
  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]
  
  def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
