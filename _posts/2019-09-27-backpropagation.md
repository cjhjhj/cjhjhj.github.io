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

### Layer 1
<img src="/assets/img/ml/nn_example_layer1.png">

(예제에서) input layer _i_ 에서 첫번째 hidden layer _j_ 로의 weight  
$$W_{ij} = \begin{bmatrix} w_{i_1j_1} & w_{i_1j_2} & w_{i_1j_3} \\ 
                           w_{i_2j_1} & w_{i_2j_2} & w_{i_2j_3} \\ 
                           w_{i_3j_1} & w_{i_3j_2} & w_{i_3j_3}
            \end{bmatrix}$$  

Hidden layer _j_ 로의 input  
$$ h1_{in1} = i_1w_{i_1j_1} + i_2w_{i_2j_1} + i_3w_{i_3j_1} + b_{j_1} $$  
$$ h1_{in2} = i_1w_{i_1j_2} + i_2w_{i_2j_2} + i_3w_{i_3j_2} + b_{j_2} $$  
...

Matrix operation으로 표시하면,  
$$ \begin{bmatrix} h1_{in1} \\ 
                  h1_{in2} \\ 
                  h1_{in3} \end{bmatrix} = 
\begin{bmatrix} w_{i_1j_1} & w_{i_2j_1} & w_{i_3j_1} \\
                w_{i_1j_2} & w_{i_2j_2} & w_{i_3j_2} \\
                w_{i_1j_3} & w_{i_2j_3} & w_{i_3j_3} \end{bmatrix} 
\begin{bmatrix} i_1 \\
                i_2 \\
                i_3 \end{bmatrix} +
\begin{bmatrix} b_{j_1} \\
                b_{j_2} \\
                b_{j_3} \end{bmatrix} = 
W_{ij}^TI + B_j $$  

Relu ($=max(0, x)$) operation을 통한 hidden layer _j_ 의 output  
$$ \begin{bmatrix} h1_{out1} \\ 
                   h1_{out2} \\ 
                   h1_{out3} \end{bmatrix} = 
\begin{bmatrix} max(0, h1_{in1}) \\ 
                max(0, h1_{in2}) \\ 
                max(0, h1_{in3}) \end{bmatrix} $$  

### Layer 2
<img src="/assets/img/ml/nn_example_layer2.png">

첫번째 hidden layer _j_ 로부터 두번째 hidden layer _k_ 로의 weight  
$$W_{jk} = \begin{bmatrix} w_{j_1k_1} & w_{j_1k_2} & w_{j_1k_3} \\ 
                           w_{j_2k_1} & w_{j_2k_2} & w_{j_2k_3} \\ 
                           w_{j_3k_1} & w_{j_3k_2} & w_{j_3k_3}
            \end{bmatrix}$$  

Hidden layer _k_ 로의 input  
$$ h2_{in1} = h1_{out1}w_{j_1k_1} + h1_{out2}w_{j_2k_1} + h1_{out3}w_{j_3k_1} + b_{k_1} $$  
$$ h2_{in2} = h1_{out1}w_{j_1k_2} + h1_{out2}w_{j_2k_2} + h1_{out3}w_{j_3k_2} + b_{k_2} $$  
...

Matrix operation으로 표시하면,  
$$ \begin{bmatrix} h2_{in1} \\ 
                  h2_{in2} \\ 
                  h2_{in3} \end{bmatrix} = 
\begin{bmatrix} w_{j_1k_1} & w_{j_2k_1} & w_{j_3k_1} \\
                w_{j_1k_2} & w_{j_2k_2} & w_{j_3k_2} \\
                w_{j_1k_3} & w_{j_2k_3} & w_{j_3k_3} \end{bmatrix} 
\begin{bmatrix} h1_{out1} \\
                h1_{out2} \\
                h1_{out3} \end{bmatrix} +
\begin{bmatrix} b_{k_1} \\
                b_{k_2} \\
                b_{k_3} \end{bmatrix} = 
W_{jk}^Th1_{out} + B_k $$  

Sigmoid ($=1 / (1+e^{-x})$) operation을 통한 hidden layer _k_ 의 output  
$$ \begin{bmatrix} h2_{out1} \\ 
                   h2_{out2} \\ 
                   h2_{out3} \end{bmatrix} = 
\begin{bmatrix} 1 / (1 + e^{-h2_{in1}}) \\ 
                1 / (1 + e^{-h2_{in2}}) \\ 
                1 / (1 + e^{-h2_{in3}}) \end{bmatrix} $$  
