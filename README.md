# GradientFlow
A mini backpropagation library - from scratch.

GradientFlow is a compact yet powerful Autograd engine designed for simplicity and learning. This tiny library, built with just around 100 lines of code, performs reverse-mode autodiff through a dynamically constructed Directed Acyclic Graph (DAG). On top of this, a lightweight neural networks library offers a PyTorch-like API in just 50 more lines. Despite its small footprint, GradientFlow can construct and train deep neural networks for binary classification, making it an excellent tool for educational purposes and experimentation.

### Features
- Autograd Engine: Implements reverse-mode autodiff over a dynamically constructed DAG. The engine is designed to operate over scalar values, which means each neuron is broken down into individual adds and multiplies.
- Tiny Neural Networks Library: Built on top of the autograd engine, this library allows you to create and train small neural networks using a PyTorch-like API.
- Educational Tool: The simplicity and compactness of the code make it an excellent resource for learning the fundamentals of backpropagation and neural networks.

### Example usage

Below is a slightly contrived example showing several possible supported operations:

```python
from gradientflow.core import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db
```

### Training a neural net

The notebook `demo.ipynb` provides a full demo of training an 2-layer neural network (MLP) binary classifier. This is achieved by initializing a neural net from `micrograd.nn` module, implementing a simple svm "max-margin" binary classification loss and using SGD for optimization. As shown in the notebook, using a 2-layer neural net with two 16-node hidden layers we achieve the following decision boundary on the moon dataset:

![2d neuron](moon_mlp.png)

### Tracing / visualization

The notebook `trace.ipynb` produces graphviz visualizations for added convenience. E.g. this one below is of a simple 2D neuron, arrived at by calling `draw` on the code below, and it shows both the data (left number in each node) and the gradient (right number in each node).

```python
from gradientflow.core import Neuron
n = Neuron(2)
x = [Value(1.0), Value(-2.0)]
y = n(x)
dot = draw(y)
```

![2d neuron](gout.svg)

### Running tests

To run the unit tests you will have to install [PyTorch](https://pytorch.org/), which the tests use as a reference for verifying the correctness of the calculated gradients. Then simply:

```bash
python -m pytest
```

Demo
Check out the python scripts and notebooks in the `test` directory to see GradientFlow in action.

Contributing
Contributions are welcome! Feel free to fork the repository and submit a pull request.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
- GradientFlow was inspired by the simplicity and elegance of PyTorch's API, and by educational resources that emphasize understanding the fundamentals of machine learning.
- GradientFlow is inspired by Andrej Karpathy's "micrograd." I’ve expanded upon the original concept by adding new features, including the implementation of the "tanh()" function and the ability to select from multiple activation functions based on user input. This library builds on the educational foundation provided by "micrograd" while offering additional flexibility for experimentation.
