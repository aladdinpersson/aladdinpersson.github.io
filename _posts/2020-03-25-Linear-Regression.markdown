---
layout: post
mathjax: true
math: true
comments: true
title:  "Derivation of Linear Regression"
excerpt: "Sometimes it is important to build a solid foundation."
date:   2020-03-25 11:00:00
---

I think linear regression is one of those methods that noone really wants to learn even though it's a very useful method in practice. It's too simple, easy, boring. And it turned to be more challenging than I initially thought. It's a clear trend to always want to learn the most fancy and shiny method available (neural nets I'm talking about you) and I want to go against this trend a little bit. I'm definitely guilty of this as much as everyone else but I've recognized it and want to spend some time on fixing it by building a solid foundation. Without further introduction let's go through two things today, first being doing linear regression by gradient descent and secondly let's understand the analytical solution to linear regression also called the normal equation.

Let's start with having our model being

$$
\hat{y}(\textbf{x}) = w_1x_1 + ... + w_n x_n = \sum_{k=1}^{n} w_k x_k =\mathbf{w}^\intercal \textbf{x},
$$

where $$\textbf{w} \in \mathcal{R}^{n \times 1}$$ and $$\textbf{x} \in \mathcal{R}^{n \times 1}$$ with $$x_1 = 1$$. We define the cost function for $$m$$ training examples or points, where we denote $$\hat{y}_{w}^{(i)}$$ is the predicted value for point $$i$$ and it's dependent on weights $$\textbf{w}$$.

$$C(\textbf{w}) = \frac{1}{m} \sum_{i=1}^{m}( \hat{y}_{w}(x^{(i)}) - y^{(i)})^2$$

The simple formula for gradient descent we want to update by doing

$$ w_j = w_j - \alpha \frac{\partial}{\partial w_{j}} C(\textbf{w}) \hspace{10pt} \forall j \in {1,2,...,n} $$

with $$\alpha$$ being the learning rate. Let's now look at solving

$$
\begin{align}
\frac{\partial}{\partial w_{j}} C(\textbf{w})  &= \frac{\partial}{\partial w_{j}}\frac{1}{m} \sum_{i=1}^{m}( \hat{y}_{w}(x^{(i)}) - y^{(i)})^2 \\
&=  \frac{1}{m} \sum_{i=1}^{m}\frac{\partial}{\partial w_{j}} \left( \sum_{k=0}^{n} w_{k}x^{(i)}_k - y^{(i)} \right) ^2 \\
&= \frac{2}{m} \sum_{i=1}^{m} \left( \sum_{k=0}^{n} w_{k}x^{(i)}_k - y^{(i)} \right) \frac{\partial}{\partial w_{j}} \left( \sum_{k=0}^{n} (w_{k}x^{(i)}_k)- y^{(i)} \right) \\
&= \frac{2}{m} \sum_{i=1}^{m} \left( \sum_{k=0}^{n} w_{k}x^{(i)}_k - y^{(i)} \right)  x_{j}^{(i)}
\end{align}
$$

When we have the update rule we simply calculate the gradient using the formula above. The pro's of utilizing gradient descent is that it's fast although the con is that it is not exact and only gives approximate optimal solutions. [Here](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/algorithms/linearregression/linear_regression_gradient_descent.py) is a link to my Github implementation. The way of finding the weights $$w$$ using an analytical formula requires some derivations. View this blog post as combination of two separate posts, as I'm using same variable names and defining them differently. We first need some notation. For a function $$f: \mathbb{R}^{m \times n} \mapsto \mathbb{R}$$ mapping from $$m \times n$$ matrices to the real numbers, we define the derivative of $$f$$ with respect to $$A$$ to be:

$$
\nabla_{A} f(A)=\left[\begin{array}{ccc}
\frac{\partial f}{\partial A_{11}} & \cdots & \frac{\partial f}{\partial A_{1 d}} \\
\vdots & \ddots & \vdots \\
\frac{\partial f}{\partial A_{n 1}} & \cdots & \frac{\partial f}{\partial A_{n d}}
\end{array}\right]
$$

Let $$\textbf{X} \in \mathcal{R}^{m \times n}, \textbf{y} \in \mathcal{R}^{m \times 1}$$ and $$\textbf{W} \in \mathcal{R}^{n \times 1}$$. We wish to minimize the cost function, and since $$\mathbf{z}^\intercal \textbf{z} = \sum_{k=0}^{n} z_{k}^{2}$$ we can write the cost as the following $$C(\textbf{w}) = \frac{1}{m} \sum_{i=1}^{m}( \hat{y}_{w}(x^{(i)}) - y^{(i)})^2$$ and remember that what we can do since the function is convex if we find a solution where the gradient equals 0 then this will be a global minimum.

$$
\begin{align}
\nabla_{W} C(\textbf{w}) &= \frac{1}{m} \sum_{i=1}^{m}( \hat{y}_{w}(x^{(i)}) - y^{(i)})^2 \\
&= \nabla_{W}  \frac{1}{m} (\textbf{XW - y})^\intercal (\textbf{XW - y}) \\
&= \nabla_{W} \frac{1}{m} (\textbf{XW})^\intercal \textbf{XW} - (\textbf{XW})^\intercal \textbf{y} -  \textbf{y}^\intercal \textbf{XW} + \textbf{y}^\intercal \textbf{y}) \\
&= \frac{1}{m}  \nabla_{W} (\textbf{XW})^\intercal \textbf{XW} - (\textbf{XW})^\intercal \textbf{y} -  \textbf{y}^\intercal \textbf{XW}) \\
&= \frac{1}{m}  \nabla_{W} (\textbf{W}\textbf{X}^\intercal  \textbf{XW} - (\textbf{XW})^\intercal \textbf{y} -  \textbf{y}^\intercal \textbf{XW}) \\
&= \frac{1}{m}  \nabla_{W} (\textbf{W}\textbf{X}^\intercal  \textbf{XW} - (\textbf{XW})^\intercal \textbf{y} -  \textbf{y}^\intercal \textbf{XW}) \\
&= \frac{1}{m}  \nabla_{W} (\textbf{W}\textbf{X}^\intercal  \textbf{XW} -  2 \textbf{y}^\intercal \textbf{XW})  \\
\end{align}
$$

Where on the last step we used that $$(\textbf{XW})^\intercal \textbf{y} =  \textbf{y}^\intercal \textbf{XW}$$ since a scalar transpose still equals the same scalar. It's not entirely obvious and can in fact be quite tricky to figure out the gradient of these matrices (you know your matrix calculus right? Because I don't..). So either when seeing these you can check wikipedia for matrix calculus identities or you can as we're gonna do try to figure them out. Let's figure out the first in the last part namely

$$  \nabla_{W} (\textbf{W}\textbf{X}^\intercal  \textbf{XW}) $$, the trick when doing matrix calculus is by converting them into summations, doing the derivative for a specific index and then converting them back to matrix form. We can define the middle part of $$ \textbf{X}^\intercal  \textbf{X}$$ as $$M$$ which is a square matrix, just to make the notation in the calculations simpler and look at the following

$$
\begin{align}
\frac{\partial \textbf{W}^\intercal M \textbf{W}}{\partial w_i} &= \frac{\partial \sum_{k=1}^{n} \sum_{j=1}^{n} w_j M_{jk} w_{k} }{\partial w_i} \\
&= \sum_{k=1}^{n} \sum_{j=1}^{n} \frac{\partial}{\partial w_i} w_j M_{jk} w_{k}  \\
&= \sum_{k=1}^{n} \sum_{j=1}^{n} \delta{ij} M_{jk} w_{k} + w_j M_{jk} \delta_{ki} \hspace{10pt} \text{Chain rule used and} \hspace{10pt} \href{https://en.wikipedia.org/wiki/Kronecker_delta}{Kronecker delta} \\
&= \sum_{k=1}^{n}  M_{ik} w_{k} +  \sum_{j=1}^{n} w_j M_{ji} \\
&= \sum_{k=1}^{n}  M_{ik} w_{k} +  \sum_{j=1}^{n} M_{ji} w_j \\
&= ((M + M^\intercal) \textbf{W})_ i
\end{align}
$$

And then if we want to have it for the entire then simply remove the i:th index, also in this case $$M$$ is a symmetric matrix (should be an easy verification) and we then find that it simplies to

$$  \nabla_{W} (\textbf{W}\textbf{X}^\intercal  \textbf{XW}) = 2\textbf{X}^\intercal \textbf{XW}$$. I won't do the second gradient it's by the same principle of converting to sum and doing the derivative for a specific index and then converting it back into matrix form but we have that $$\nabla_{W} (2 \textbf{y}^\intercal \textbf{XW}) = 2 \textbf{X}^\intercal \textbf{y}$$. Hence going back we now have

$$
\begin{align}
\nabla_{W} C(\textbf{w}) &= \frac{1}{m}( 2\textbf{X}^\intercal \textbf{XW} - 2 \textbf{X}^\intercal \textbf{y}) \hspace{5pt} \text{Set to 0 for global min} \\
&= 0
\end{align}
$$

This gives us

$$ \textbf{X}^\intercal \textbf{XW} = \textbf{X}^\intercal \textbf{y} $$

And we then want $$\textbf{W}$$ which then gives us

$$ \textbf{W} = (\textbf{X}^\intercal \textbf{X})^{-1} \textbf{X}^\intercal \textbf{y} $$

And everything is sweet, happy and life again makes sense. [Here](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/algorithms/linearregression/linear_regression_normal_equation.py) is my implementation on Github for using this analytical solution. The pro of this is it gives the exact optimal solution, but it's slow compared to gradient descent.
