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
$$h_{in} = x \cdot W^1 = 
\begin{bmatrix}
  x_1 & x_2 & x_3 & x_4 
\end{bmatrix}
\begin{bmatrix}
  w^1_{11} & w^1_{12} & \cdots & w^1_{1,10} \\
  w^1_{21} & w^1_{22} & \cdots & w^1_{2,10} \\
  \vdots & \vdots & \ddots & \vdots \\
  w^1_{41} & w^1_{42} & \cdots & w^1_{4,10}
\end{bmatrix} = 
\begin{bmatrix}
  h_{in_1} & h_{in_2} & \cdots & h_{in_{10}}
\end{bmatrix}$$  
$$h_{out} = ReLU(h_{in}) = 
\begin{bmatrix}
  max(0, h_{in_1}) & max(0, h_{in_2}) & \cdots & max(0, h_{in_{10}})
\end{bmatrix}$$  
$$o_{in} = h_{out} \cdot W^2 =
\begin{bmatrix}
  h_{out_1} & \cdots & h_{out_{10}} 
\end{bmatrix}
\begin{bmatrix}
  w^2_{11} & w^2_{12} & w^2_{13} \\
  w^2_{21} & w^2_{22} & w^2_{23} \\
  \vdots & \ddots & \vdots \\
  w^2_{10,1} & w^2_{10,2} & w^2_{10,3}
\end{bmatrix} = 
\begin{bmatrix}
  o_{in_1} & o_{in_2} & o_{in_3}
\end{bmatrix}$$  

Let softmax(x) function be $p(x)$,  
$$o_{out} = softmax(o_{in}) = p(o_{in}) =
\begin{bmatrix}
  p(o_{in_1}) & p(o_{in_2}) & p(o_{in_3})
\end{bmatrix} = 
\begin{bmatrix}
  \frac{e^{o_{in_1}}}{\sum_{k = 1}^3 e^{o_{in_k}}} &
  \frac{e^{o_{in_2}}}{\sum_{k = 1}^3 e^{o_{in_k}}} &
  \frac{e^{o_{in_3}}}{\sum_{k = 1}^3 e^{o_{in_k}}}
\end{bmatrix}$$  

Softmax loss function is defined as,  
$$L = \sum_{k = 1}^3 L_k = \sum_{k = 1}^3 o_{out_k} = o_{out_1} + o_{out_2} + o_{out_3}$$  

### Back-propagation
First, let's think about the update of $W^2$ matrix (i.e. back-progation between output and hidden layers). According to the chain rule,
the derivative of loss w.r.t $w^2_{11}$ can be expressed as follows,  
$$\frac{\partial L}{\partial w^2_{11}} = \frac{\partial L}{\partial o_{out_1}}\frac{\partial o_{out_1}}{\partial o_{in_1}}\frac{\partial o_{in_1}}{\partial w^2_{11}}$$  

The first derivative is  
$$\frac{\partial L}{\partial o_{out_1}} = \frac{\partial}{\partial o_{out_1}} (o_{out_1} + o_{out_2} + o_{out_3}) = 1$$  

The second derivative is  
$$\frac{\partial o_{out_1}}{\partial o_{in_1}} = 
\begin{bmatrix}
  \frac{\partial p(o_{in_1})}{\partial o_{in_1}} & 
  \frac{\partial p(o_{in_2})}{\partial o_{in_1}} & 
  \frac{\partial p(o_{in_3})}{\partial o_{in_1}}
\end{bmatrix} = 
\frac{\partial p(o_{in_1})}{\partial o_{in_1}} = 
p(o_{in_1})(1 - p(o_{in_1}))$$ 
(The proof can be found below)

The third derivative is  
$$\frac{\partial o_{in_1}}{\partial w^2_{11}} = 
\frac{\partial}{\partial w^2_{11}} ( h_{out_1}w^2_{11} + h_{out_2}w^2_{21}+ \cdots + h_{out_{10}}w^2_{10,1}) = 
h_{out_1}$$  

Generlized results are as follows,  
$$\frac{\partial L}{\partial w^2_{jk}} = \frac{\partial L}{\partial o_{out_k}}\frac{\partial o_{out_k}}{\partial o_{in_k}}\frac{\partial o_{in_k}}{\partial w^2_{jk}}$$  
$$\frac{\partial L}{\partial o_{out_k}} = \frac{\partial}{\partial o_{out_k}} (o_{out_1} + \cdots + o_{out_k} + \cdots) = 1$$  
$$\frac{\partial o_{out_k}}{\partial o_{in_k}} = 
\begin{bmatrix}
  \frac{\partial p(o_{in_1})}{\partial o_{in_k}} &
  \cdots &
  \frac{\partial p(o_{in_k})}{\partial o_{in_k}} & 
  \cdots
\end{bmatrix} = 
\frac{\partial p(o_{in_k})}{\partial o_{in_k}} = 
p(o_{in_k})(1 - p(o_{in_k}))$$  
$$\frac{\partial o_{in_k}}{\partial w^2_{jk}} = 
\frac{\partial}{\partial w^2_{jk}} ( h_{out_1}w^2_{1k} + \cdots + h_{out_j}w^2_{jk}+ \cdots + h_{out_{10}}w^2_{10,k}) = 
h_{out_j}$$  
$$\therefore \frac{\partial L}{\partial w^2_{jk}} = 
\frac{\partial L}{\partial o_{out_k}}\frac{\partial o_{out_k}}{\partial o_{in_k}}\frac{\partial o_{in_k}}{\partial w^2_{jk}} =
1\cdot p(o_{in_k})(1 - p(o_{in_k}))\cdot h_{out_j}$$  

The derivative of $W^2$ can be expressed as,  
$$\begin{aligned}
\frac{\partial L}{\partial W^2} &= 
\begin{bmatrix}
  \frac{\partial L}{\partial w^2_{11}} & \frac{\partial L}{\partial w^2_{12}} & \frac{\partial L}{\partial w^2_{13}} \\
  \frac{\partial L}{\partial w^2_{21}} & \frac{\partial L}{\partial w^2_{22}} & \frac{\partial L}{\partial w^2_{13}} \\
  \vdots & \vdots & \vdots \\
  \frac{\partial L}{\partial w^2_{10,1}} & \frac{\partial L}{\partial w^2_{10,2}} & \frac{\partial L}{\partial w^2_{10,3}}
\end{bmatrix} = 
\begin{bmatrix}
  h_{out_1}\cdotp(o_{in_1})(1 - p(o_{in_1})) & h_{out_1}\cdotp(o_{in_2})(1 - p(o_{in_2})) & h_{out_1}\cdotp(o_{in_3})(1 - p(o_{in_3})) \\
  h_{out_2}\cdotp(o_{in_1})(1 - p(o_{in_1})) & h_{out_2}\cdotp(o_{in_2})(1 - p(o_{in_2})) & h_{out_2}\cdotp(o_{in_3})(1 - p(o_{in_3})) \\
  \vdots & \vdots & \vdots \\
  h_{out_{10}}\cdotp(o_{in_1})(1 - p(o_{in_1})) & h_{out_{10}}\cdotp(o_{in_2})(1 - p(o_{in_2})) & h_{out_{10}}\cdotp(o_{in_3})(1 - p(o_{in_3}))
\end{bmatrix} \\
&= 
\begin{bmatrix}
  h_{out_1} \\
  h_{out_2} \\
  \vdots \\
  h_{out_{10}}
\end{bmatrix}
\begin{bmatrix}
  p(o_{in_1})(1 - p(o_{in_1})) & p(o_{in_2})(1 - p(o_{in_2})) & p(o_{in_3})(1 - p(o_{in_3}))
\end{bmatrix}
\end{aligned}$$


Next, we need to go up to $W^1$ which makes the relationship between input an hidden layer. As similar to above, the derivative of loss w.r.t $w^1_{11}$ can be expressed as follows based on chain rule,  
$$\frac{\partial L}{\partial w^1_{11}} = \frac{\partial L}{\partial h_{out_1}}\frac{\partial h_{out_1}}{\partial h_{in_1}}\frac{\partial h_{in_1}}{\partial w^1_{11}}$$  

The last derivative is simple as follows,  
$$\frac{\partial h_{in_1}}{\partial w^1_{11}} = \frac{\partial}{\partial w^1_{11}} (x_1 \cdot w^1_{11} + \cdots + x_4 \cdot w^1_{41}) = x_1$$  

The second derivative is also simple (the derivative of ReLU function),  
$$\begin{aligned}
  \frac{\partial h_{out_1}}{\partial h_{in_1}} = \frac{\partial}{\partial h_{in_1}} max(0, h_{in_1}) &= 
  0 \ (h_{in_1} \le 0) \\
  &= 1 \ (h_{in_1} \gt 0)
\end{aligned}$$  

The first derivative is a bit complicated since $h_{out_1}$ contributes every $L_1$, $L_2$ and $L_3$.  
$$\frac{\partial L}{\partial h_{out_1}} = \frac{\partial (L_1 + L_2 + L_3)}{\partial h_{out_1}}$$  

Each derivative is as follows,  
$$\frac{\partial L_1}{\partial h_{out_1}} = \frac{\partial L_1}{\partial o_{out_1}}\frac{\partial o_{in_1}}{\partial o_{out_1}}\frac{\partial o_{in_1}}{\partial h_{out_1}} = 1 \cdot p(o_{in_1})(1 - p(o_{in_1})) \cdot w^2_{11}$$  
$$\frac{\partial L_2}{\partial h_{out_1}} = \frac{\partial L_2}{\partial o_{out_2}}\frac{\partial o_{in_2}}{\partial o_{out_2}}\frac{\partial o_{in_2}}{\partial h_{out_1}} = 1 \cdot p(o_{in_2})(1 - p(o_{in_2})) \cdot w^2_{12}$$  
$$\frac{\partial L_3}{\partial h_{out_1}} = \frac{\partial L_3}{\partial o_{out_3}}\frac{\partial o_{in_3}}{\partial o_{out_3}}\frac{\partial o_{in_3}}{\partial h_{out_1}} = 1 \cdot p(o_{in_3})(1 - p(o_{in_3})) \cdot w^2_{13}$$  
$$\frac{\partial L}{\partial h_{out_1}} = w^2_{11}p(o_{in_1})(1 - p(o_{in_1})) + 
w^2_{12}p(o_{in_2})(1 - p(o_{in_2})) +
w^2_{13}p(o_{in_3})(1 - p(o_{in_3}))$$  

Therefore,  
$$\begin{aligned}
  \frac{\partial L}{\partial w^1_{11}} &= x_1(w^2_{11}p(o_{in_1})(1 - p(o_{in_1})) + w^2_{12}p(o_{in_2})(1 - p(o_{in_2})) + w^2_{13}p(o_{in_3})(1 - p(o_{in_3}))) \ (h_{in_1} > 0) \\
  &= 0 \ (otherwise)
\end{aligned}$$  

Similarly,  
$$\begin{aligned}
  \frac{\partial L}{\partial w^1_{12}} &= x_1(w^2_{11}p(o_{in_1})(1 - p(o_{in_1})) + w^2_{12}p(o_{in_2})(1 - p(o_{in_2})) + w^2_{13}p(o_{in_3})(1 - p(o_{in_3}))) \ (h_{in_1} > 0) \\
  &= 0 \ (otherwise)
\end{aligned}$$  

Let $p(o_{in_k})(1 - p(o_{in_k})$ be $q_k$ for simplicity. Then, the generalized result is,  
$$\begin{aligned}
  \frac{\partial L}{\partial w^1_{ij}} &= x_i(w^2_{j1}q_1 + w^2_{j2}q_2 + w^2_{j3}q_3)  = x_i\sum_k (w^1_{jk}q_k) \ (h_{in_1} > 0) \\
  &= 0 \ (otherwise)
\end{aligned}$$  

$$\begin{aligned}
	\begin{bmatrix}
		\frac{\partial L}{\partial w^1_{11}} & \frac{\partial L}{\partial w^1_{12}} & \cdots & 
		\frac{\partial L}{\partial w^1_{1,10}} \\
		\frac{\partial L}{\partial w^1_{21}} & \frac{\partial L}{\partial w^1_{22}} & \cdots & 
		\frac{\partial L}{\partial w^1_{2,10}} \\
		\vdots & \vdots & \ddots & \vdots \\
		\frac{\partial L}{\partial w^1_{41}} & \frac{\partial L}{\partial w^1_{42}} & \cdots & 
		\frac{\partial L}{\partial w^1_{4,10}}
	\end{bmatrix} & = 
	\begin{bmatrix}
		x_1(w^2_{11}q_1 + w^2_{12}q_2 + w^2_{13}q_3) & x_1(w^2_{21}q_1 + w^2_{22}q_2 + 
		w^2_{23}q_3) & \cdots & x_1(w^2_{10,1}q_1 + w^2_{10,2}q_2 + w^2_{10,3}q_3) \\
		x_2(w^2_{11}q_1 + w^2_{12}q_2 + w^2_{13}q_3) & x_2(w^2_{21}q_1 + w^2_{22}q_2 + 
		w^2_{23}q_3) & \cdots & x_2(w^2_{10,1}q_1 + w^2_{10,2}q_2 + w^2_{10,3}q_3) \\
		\vdots & \vdots & \ddots & \vdots \\
		x_4(w^2_{11}q_1 + w^2_{12}q_2 + w^2_{13}q_3) & x_4(w^2_{21}q_1 + w^2_{22}q_2 + 
		w^2_{23}q_3) & \cdots & x_4(w^2_{10,1}q_1 + w^2_{10,2}q_2 + w^2_{10,3}q_3)
	\end{bmatrix} \\
	&=
	\begin{bmatrix}
		x_1 \\ x_2 \\ x_3 \\ x_4
	\end{bmatrix}
	\left(
		\begin{bmatrix}
			q_1 & q_2 & q_3
		\end{bmatrix}
		\begin{bmatrix}
			w^2_{11} & \cdots & w^2_{13} \\
			w^2_{21} & \cdots & w^2_{23} \\
			\vdots & \vdots & \vdots \\
			w^2_{10,1} & \cdots & w^2_{10,3}
		\end{bmatrix} ^ T
	\right)
\end{aligned}$$  
