# AUDIT & FIX: neural-network-basic

## CRITIQUE
- **Generally Sound**: This project is a well-known micrograd clone and the technical content is largely correct.
- **Missing Optimizer Abstraction**: The audit correctly notes there's no Optimizer class. Separating the optimization logic (SGD, learning rate, weight decay) from the training loop is both good software design and pedagogically important for understanding how PyTorch optimizers work.
- **Activation Function on Last Layer**: M3 pitfall says 'Missing activation on last layer' but for regression tasks, the last layer should NOT have an activation function. For binary classification, it should be sigmoid. The pitfall is misleading without specifying the task type.
- **No Numerical Gradient Verification**: There's no requirement to verify analytical gradients against numerical gradients (finite differences). This is the standard debugging technique for autograd implementations and is critical for correctness.
- **Division Operation Incomplete**: M1 says 'Value supports... division' but implementing division properly requires handling the gradient for a/b = a * b^(-1), which chains through power and multiply. This subtlety should be noted.
- **Tanh Backward Not Specified**: The derivative of tanh is 1 - tanh²(x), and this must be implemented in the backward pass. It's implicitly covered but should be explicitly tested.
- **Training Loop Dataset Too Trivial**: M4 doesn't specify what dataset to train on. Without a concrete task (e.g., learning XOR, or fitting a simple function), the milestone is underspecified.
- **Missing: Graph Visualization**: Learning outcome mentions 'Visualize computational graphs' but no milestone requires it.

## FIXED YAML
```yaml
id: neural-network-basic
name: "Neural Network from Scratch (micrograd)"
description: "Build a scalar-valued autograd engine and train a small neural network, implementing backpropagation from first principles."
difficulty: intermediate
estimated_hours: "18-28"
essence: >
  Scalar-valued automatic differentiation through dynamic computational graph
  construction, reverse-mode gradient propagation via topological sorting, and
  operator overloading to mechanically apply the chain rule—culminating in a
  trainable multi-layer perceptron.
why_important: >
  Building this demystifies how PyTorch and TensorFlow work under the hood,
  revealing that the core differentiation engine is surprisingly simple (~100 lines)
  while teaching the mathematical foundations essential for debugging and
  optimizing neural networks.
learning_outcomes:
  - Implement reverse-mode automatic differentiation using computational graphs
  - Build a scalar autograd engine with operator overloading for arithmetic operations
  - Design backpropagation through arbitrary computation graphs using topological sorting
  - Verify analytical gradients against numerical gradients for correctness
  - Implement neural network primitives (Neuron, Layer, MLP) from first principles
  - Build a training loop with loss computation, backward pass, and optimizer step
  - Visualize computational graphs to trace derivative calculations
skills:
  - Automatic Differentiation
  - Computational Graphs
  - Backpropagation
  - Gradient Descent
  - Neural Network Architecture
  - Chain Rule Implementation
  - Topological Sort
  - Operator Overloading
tags:
  - ai-ml
  - backprop
  - autograd
  - intermediate
  - python
  - neural-networks
architecture_doc: architecture-docs/neural-network-basic/index.md
languages:
  recommended:
    - Python
  also_possible:
    - Julia
    - JavaScript
resources:
  - name: "Micrograd Repository (Karpathy)"
    url: https://github.com/karpathy/micrograd
    type: repository
  - name: "Micrograd Video Tutorial (Karpathy)"
    url: https://www.youtube.com/watch?v=VMj-3S1tku0
    type: video
  - name: "Calculus on Computational Graphs: Backpropagation"
    url: https://colah.github.io/posts/2015-08-Backprop/
    type: article
prerequisites:
  - type: skill
    name: "Calculus (derivatives, chain rule)"
  - type: skill
    name: "Python (classes, operator overloading via __add__, __mul__)"
  - type: skill
    name: "Basic linear algebra (dot product, matrix concepts)"
milestones:
  - id: neural-network-basic-m1
    name: "Value Class with Autograd"
    description: >
      Create a Value class that wraps scalars, tracks the computational graph,
      and supports arithmetic operations with gradient tracking.
    acceptance_criteria:
      - "Value class wraps a Python float and stores a .grad field initialized to 0.0"
      - "Each Value tracks its child operands (_children) and the operation that produced it (_op)"
      - "Operator overloads support: addition (+), multiplication (*), power (**), negation (-), subtraction (-), and true division (/)"
      - "Reverse operators (__radd__, __rmul__) handle expressions like 2 + Value(3) where the scalar is on the left"
      - "Division is implemented as a * b**(-1), chaining through the power operation's backward"
      - "Each operation stores a _backward closure that computes local gradients and accumulates them into children's .grad fields"
      - "Gradient accumulation uses += (not =) to correctly handle the case where a Value is used as input to multiple operations"
    pitfalls:
      - "Using = instead of += for gradient accumulation silently drops gradients when a value feeds into multiple operations"
      - "Not implementing __radd__/__rmul__ causes TypeError when doing int + Value or float * Value"
      - "Division backward: d/da(a/b) = 1/b and d/db(a/b) = -a/b²—easy to get the sign wrong"
      - "Power backward must handle the case where the exponent is negative (e.g., b**(-1) for division)"
    concepts:
      - Computational graphs
      - Operator overloading
      - Gradient accumulation
      - Local derivative computation
    skills:
      - Python dunder methods (__add__, __mul__, etc.)
      - Closure-based backward functions
      - Computational graph data structure design
    deliverables:
      - "Value class with scalar data, gradient field, children tracking, and operation label"
      - "Arithmetic operator overloads (+, *, **, -, /) with reverse operator support"
      - "Per-operation _backward closures computing local gradients with accumulation"
      - "String representation (__repr__) showing data and gradient values for debugging"
    estimated_hours: "4-6"

  - id: neural-network-basic-m2
    name: "Backward Pass & Gradient Verification"
    description: >
      Implement the backward() method using topological sort for reverse-mode
      automatic differentiation, and verify correctness with numerical gradients.
    acceptance_criteria:
      - "Topological sort produces a valid ordering of all Values in the graph from inputs to output using DFS with visited set"
      - "backward() sets the output Value's gradient to 1.0, then iterates in reverse topological order calling each node's _backward"
      - "Gradients are correct for a multi-operation expression: e.g., for L = (a*b + c)**2, verify dL/da, dL/db, dL/dc analytically"
      - "Numerical gradient verification: for each parameter, compute (f(x+h) - f(x-h)) / 2h and verify it matches the analytical gradient within 1e-5 relative tolerance"
      - "Calling backward() twice without zeroing gradients accumulates (doubles) gradients—this behavior is documented and tested"
      - "Tanh activation function is implemented on Value with correct backward: grad = (1 - tanh²(x)) * upstream_grad"
      - "ReLU activation function is implemented on Value with correct backward: grad = (1 if x > 0 else 0) * upstream_grad"
    pitfalls:
      - "Wrong topological order: processing a node before its consumers causes zero gradients"
      - "Not zeroing gradients before a new backward pass causes incorrect accumulated gradients from previous iterations"
      - "Tanh backward must use the output value (tanh(x)), not recompute from input—avoids numerical instability"
      - "Numerical gradient step size h too large gives inaccurate approximation; too small causes floating point errors—use h=1e-6"
    concepts:
      - Reverse-mode automatic differentiation
      - Topological sort (DFS-based)
      - Chain rule mechanical application
      - Numerical gradient checking
    skills:
      - Graph traversal algorithms (DFS)
      - Gradient computation for elementary operations
      - Numerical verification methodology
      - Activation function implementation
    deliverables:
      - "Topological sort using DFS with visited set over the computational graph"
      - "backward() method propagating gradients in reverse topological order"
      - "Tanh and ReLU activation functions with correct backward implementations"
      - "Numerical gradient checker comparing analytical vs. finite-difference gradients"
    estimated_hours: "3-5"

  - id: neural-network-basic-m3
    name: "Neuron, Layer, and MLP"
    description: >
      Build neural network components (Neuron, Layer, MLP) using the Value class,
      with proper weight initialization and parameter collection.
    acceptance_criteria:
      - "Neuron computes: activation(Σ(w_i * x_i) + b) where weights and bias are Value objects initialized with random values in [-1, 1]"
      - "Neuron supports configurable activation function: tanh for hidden layers, linear (no activation) for output layer"
      - "Layer groups N neurons and returns a list of outputs (or single Value if N=1) for a given input vector"
      - "MLP chains multiple Layers with configurable sizes (e.g., MLP(3, [4, 4, 1]) = 3 inputs, two hidden layers of 4, 1 output)"
      - "parameters() method recursively collects all trainable Value objects (weights and biases) from the entire MLP hierarchy"
      - "Parameter count matches expected: for MLP(3, [4, 4, 1]), parameters = (3*4+4) + (4*4+4) + (4*1+1) = 33"
    pitfalls:
      - "Applying activation function on the output layer for regression tasks produces bounded outputs—use linear (identity) for output"
      - "parameters() missing some weights causes those weights to never update during training"
      - "Weight initialization with all zeros causes symmetry: all neurons compute the same thing and learn identically"
      - "Returning a list vs. single Value from Layer causes inconsistent types downstream—handle the N=1 case explicitly"
    concepts:
      - Neural network architecture (feed-forward)
      - Weight initialization
      - Activation functions and their purpose
      - Parameter collection for optimization
    skills:
      - Object-oriented design for ML components
      - Recursive parameter collection
      - Forward pass implementation
      - Architecture configuration
    deliverables:
      - "Neuron class with weights, bias, configurable activation, and forward computation"
      - "Layer class grouping N neurons with collective forward pass"
      - "MLP class chaining layers with configurable architecture"
      - "parameters() method returning all trainable Values for the network"
    estimated_hours: "3-5"

  - id: neural-network-basic-m4
    name: "Training Loop & Optimizer"
    description: >
      Build a training loop with loss computation, backward pass, gradient zeroing,
      and an SGD optimizer to train the MLP on a concrete task.
    acceptance_criteria:
      - "Training dataset: a simple classification or regression task (e.g., learn a moon dataset or XOR-like function with at least 20 data points)"
      - "MSE loss is computed as the mean of (prediction - target)² over all training examples"
      - "SGD optimizer class encapsulates: zero_grad() to reset all parameter gradients, and step() to update parameters by p.data -= lr * p.grad"
      - "Training loop repeats: forward pass → compute loss → backward() → optimizer.step() → optimizer.zero_grad() for each epoch"
      - "Loss decreases monotonically (on average) over 100+ epochs, demonstrating that the network is learning"
      - "Final loss is below 0.01 on the training data for the chosen task, confirming convergence"
      - "Loss vs. epoch plot shows the convergence curve"
    pitfalls:
      - "Forgetting zero_grad() causes gradient accumulation across epochs, leading to divergence"
      - "Learning rate too high (>0.1) causes loss oscillation or explosion; too low (<0.0001) causes negligible progress"
      - "Not enough training epochs stops before convergence—run until loss plateaus"
      - "Modifying p.data directly (not through Value arithmetic) is necessary to avoid adding update operations to the computational graph"
    concepts:
      - Training loop structure
      - Stochastic gradient descent
      - Loss function design
      - Optimizer abstraction
    skills:
      - Optimization loop implementation
      - Hyperparameter tuning (learning rate, epochs)
      - Loss monitoring and convergence detection
      - Clean separation of optimizer from training logic
    deliverables:
      - "SGD optimizer class with zero_grad() and step(learning_rate) methods"
      - "MSE loss function computing mean squared error over training examples"
      - "Training loop executing forward → loss → backward → step → zero_grad for configurable epochs"
      - "Loss convergence plot showing loss vs. epoch number"
      - "Trained MLP demonstrating correct predictions on the training task"
    estimated_hours: "4-6"
```