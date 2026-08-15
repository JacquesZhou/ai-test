```markdown
# MLP Notes

## 1. What is an MLP?

MLP means Multilayer Perceptron.

A typical MLP consists of linear transformations and nonlinear
activation functions.

For this model:

x -> Linear -> ReLU -> Linear -> ReLU -> Linear -> y

Mathematically:

F(x) = W3 ReLU(W2 ReLU(W1 x + b1) + b2) + b3

## 2. Linear Layer

A linear layer performs:

z = Wx + b

The parameters W and b are learned during training.

## 3. Why ReLU?

Without activation functions, multiple linear layers are still equivalent
to one linear transformation.

ReLU is defined as:

ReLU(x) = max(0, x)

It gives the network the ability to approximate nonlinear functions.

## 4. Loss

This example uses Mean Squared Error:

L = mean((prediction - target)^2)

The goal of training is to minimize L.

## 5. Backpropagation

loss.backward()

uses automatic differentiation and the chain rule to calculate gradients:

dL/dW

and

dL/db

## 6. Optimizer

optimizer.step()

uses these gradients to update the parameters.

The entire learning process is:

input
-> forward
-> prediction
-> loss
-> backward
-> gradient
-> optimizer
-> new parameters