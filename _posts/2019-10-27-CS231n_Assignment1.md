---
layout: post
title: "CS231n Assignment 1"
categories: ml
comments: true
---

Assignment #1 of Stanford CS231n

### Two-layer network
A two-layer fully-connected neural network. The net has an input dimension of N, a hidden layer dimension of H, 
and performs classification over C classes. We train the network with a softmax loss function and L2 regularization on the
weight matrices. The network uses a ReLU nonlinearity after the first fully connected layer.


In other words, the network has the following architecture:  
_input - fully connected layer - ReLU - fully connected layer - softmax_  
The outputs of the second fully-connected layer are the scores for each class.

From "two_layer_net.ipynb", the conditions are as follows,
- input_size = 4
- hidden_size = 10
- num_classes = 3
- num_inputs = 5

Figure


Softmax loss function is defined as follows.  
$$L = \sum_i L_i = \sum_i \left(\frac{e^{o_{in_i}}}{\sum_j e^{o_{in_j}}}\right)$$  

It means that (for one input sample), $x$ = [1 x 4] vector, $W^1$ = [4 x 10] matrix, $W^2$ = [10 x 3] matrix, 
$h_{in}$ and $h_{out}$ = [1 x 10] vector, and $o_{in}$ and $o_{out}$ = [1 x 3] vector, i.e. scores for three classes.  

### Forward-propagation
$$\begin{aligned}
h_{in} & = x \cdot W^1 = 
\begin{bmatrix}
  x_1 & x_2 & x_3 & x_4 
\end{bmatrix}
\begin{bmatrix}
  w^1_{11} & w^1_{12} & \cdots & w^1_{1,10} \\
  w^1_{21} & w^1_{22} & \cdots & w^1_{2,10} \\
  \vdots & \vdots & \ddots & \vdots \\
  w^1_{41} & w^1_{42} & \cdots & w^1_{4,10}
\end{bmatrix} \\
& = 
\begin{bmatrix}
  h_{in_1} & h_{in_2} & \cdots & h_{in_{10}}
\end{bmatrix} 
\end{aligned}$$   
$$h_{out} = ReLU(h_{in}) = 
\begin{bmatrix}
  max(0, h_{in_1}) & max(0, h_{in_2}) & \cdots & max(0, h_{in_{10}})
\end{bmatrix}$$

test2
