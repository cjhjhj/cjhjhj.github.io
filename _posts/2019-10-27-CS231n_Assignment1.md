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

<figure>  


Softmax loss function is defined as follows,  

$$L = \sum_i L_i = \sum_i \left(\frac{e^{o_{in_i}}}{\sum_j e^{o_{in_j}}}\right)$$  

It means that (for one input sample), $x$ = [1 x 4] vector, $W^1$ = [4 x 10] matrix, $W^2$ = [10 x 3] matrix, 
$h_{in}$ and $h_{out}$ = [1 x 10] vector, and $o_{in}$ and $o_{out}$ = [1 x 3] vector, i.e. scores for three classes.  

### Forward-propagation

### Back-propagation
