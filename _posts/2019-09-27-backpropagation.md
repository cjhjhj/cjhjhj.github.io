---
layout: post
title: "Backpropagation"
categories: ml
comments: true
---

Backpropagation을 구현하기 위해서 알고리즘을 수학적으로 리뷰해보고 python으로 구현하기 위해 정리를 해본다. 이 정리는 다음 웹사이트 (https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c)의 내용을 기반으로 작성되었다.  
일반적인 neural network의 구조는 다음 그림과 같다.

<img src="/assets/img/ml/nn_example.png">

전체적인 알고리즘과 backpropagation의 수식을 유도하기 위해서는 먼저 forward propagation을 살펴보아야 한다. 
먼저 input으로부터 첫번째 hidden layer의 input h1<sub>_in1_</sub>을 구하고, activation function (이 예제에서는 첫번째 layer에 Relu를 사용)을 거쳐 hidden layer의 output h1<sub>_out1_</sub>을 구하는 과정을 살펴보자.

<img src="/assets/img/ml/nn_example_layer1.png">

예제에서 input layer _i_ 에서 첫번째 hidden layer _j_ 로의 weight들은 다음과 같은 W<sub>ij</sub> matrix로 정의된다.
$$a = b + c$$
$$W_ij = \begin{matrix} w_i1j1 & w_i1j2 & w_i1j3 \\ w_i2j1 & w_i2j2 & w_i2j3 \\ w_i3j1 & w_i3j2 & w_i3j3 \\ \end{matrix}$$

