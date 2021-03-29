---
layout: post
mathjax: true
math: true
comments: true
title:  "Understanding formulas for backpropogation"
excerpt: "Derivation for backpropogation for a 2-layered Neural Network with a Softmax classifier"
date:   2018-09-20 11:00:00
---

After doing the derivations for a simple 2 layered neural network only one month ago and when faced with the problem again it took me quite to figure it out. After some reflecting I realized it's better to start documenting things that I learn. Primarily it's so that it will be easy to go back if I am wondering about this problem in the future but also there might be someone that is on a similar journey as mine and can be benefited by seeing how someone else solved this problem. So without further introduction let's try to figure out how the derivations will look like. I will go through how I derive the formulas for backpropogation for a simple 2-layered neural network and also how I implement this neural network from scratch in numpy. First of all the notation varies a lot, I will try to use notation that I have learned from Andrew Ngs courses which if we quickly go through them is that superscript ($$i$$) denotes the $$i^{th}$$ training example while superscript $$[l]$$ in brackets will denote the $$l^{th}$$ layer. capital $$L$$ will be the total amount of layers in the network which in this case will just be 2. Any other notation I will try to make clear when I use them.

When thinking about backpropogation the best way to think about it is through a computational graph. For this two layered neural network it would in simplified terms look like this

<div class="imgcap">
<img src="/assets/derivbackprop/ffdiagram.png" style="border:none; width:80%;">
</div>

Where $$ Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]}$$ and $$ A^{[l]} = ReLU(Z^{[l]}) $$ except for $$A^{[0]}$$ which denotes the input. After performing the forward pass of the neural network, one needs to go backward in the computational graph to change the current parameters of the network so that the network actually learns and improves for each training example. This is really what backpropogation is about. The loss for this computational graph is to take the negative log of the softmax classifier on our last node before the loss which in this case is $$Z^{[2]}$$. The classifier function looks like

$$
\begin{align}
f_{j} = \frac{e^{Z^{[2]}_{j}}}{\sum_{c=1}^{C} e^{Z^{[2]}_{c}}} \hspace{0.5in} &\text{Softmax classifier} \\
\end{align}
$$

where the $$f_{j}$$ stands for the class score for the $$j$$-th element in one training example in the matrix $$Z^{[2]}$$ and $$C$$ stands for the amount of classes. To calculate the loss we use the cross-entropy loss which takes the negative loss of our prediction $$\hat{y}$$ for the correct label of that particular training example $$i$$ which we denote $$y_{i}$$. The loss will therefore look like

$$
\begin{align}
L_i = - \frac{1}{m} \log\left(\frac{e^{Z^{[2]}_{y_i}}}{ \sum_{c=1}^{C} e^{Z^{[2]}_{c}} }\right) \hspace{0.5in} \text{or equivalently} \hspace{0.5in} L_{i} = - \frac{1}{m} \Big(Z_{y_i} + \log\sum_{c=1}^{C} e^{Z^{[2]}_{c}} \Big)\\
\end{align}
$$

It could be a bit tricky to see how these are equivalent but really it is just by using the logarithm rule that

$$ log \Big(\frac{X}{Y}\Big) = log(X) - log(Y).$$

Now the idea is to look that, from the computational graph we want to figure out how the previous node impacts the loss. Remember that $$Z^{[2]}$$ is a matrix containing all connections from previous nodes and connections to the following nodes in the computational graph. The first picture showing the computational graph is a useful one, but it is quite simplified and more compact as all variables $$Z^{[1]}, A^{[1]}, ... $$ in it are matrices. How it actually looks like would be more similar to

<div class="imgcap">
<img src="/assets/derivbackprop/fullnetwork.png" style="border:none; width:80%;">
<div class="thecap">Image credit: Couldn't find creator.</div>
</div>

Where the hidden layer 3 can in our case be viewed as the $$Z^{[2]}$$ that contains previous connections from hidden layer 2 and connections to the output layer. To find the derivative we therefore wish to know how a specific node $$k$$ out of  $$Z^{[2]}$$ which we will denote $$Z_{k}^{[2]}$$ impacts the output layer. This can be written mathematically as


$$
\begin{align}
\frac{\partial{Loss}}{\partial{Z_{k}^{[2]}}} &= \\
&= \frac{1}{m} \cdot  \Big( -\frac{\partial}{\partial{Z_{k}^{[2]}}} Z_{y_{i}}^{[2]} + \frac{\partial}{\partial{Z_{k}^{[2]}}}  \log\sum_{c=1}^{C} e^{Z^{[2]}_{c}} \Big) \\
&= \frac{1}{m} \cdot \Big(-\delta_{k, y_{i}} + \frac{1}{\sum_{c=1}^{C} e^{Z^{[2]}_{c}}} \cdot e^{Z_{k}^{[2]}}\Big) \hspace{0.5in} \href{https://en.wikipedia.org/wiki/Kronecker_delta}{Kronecker delta} \\
&=\frac{1}{m} \cdot \Big(\frac{e^{Z_{k}^{[2]}}}{\sum_{c=1}^{C} e^{Z^{[2]}_{c}}} -\delta_{k, y_{i}}\Big) \hspace{0.5in} \text{Recognize the first term is just the softmax classifier} \\
\\
\tag{1}
&= \frac{1}{m} \cdot \big( f_{k} -\delta_{k, y_{i}} \big)
\label{eqn:firstbackprop}
\end{align}
$$

We've now found the first calculation for how the $$Z^{[2]}$$ influences the loss or output layer. From our simplified computational graph we can visualize this as

<div class="imgcap">
<img src="/assets/derivbackprop/backprop1.png" style="border:none; width:80%;">
</div>

We now realize that there are variables that impact $$Z^{[2]}$$ that in turn impacts the classifier which makes the loss higher or lower. So we need to go backward and tune the variables that change $$Z^{[2]}$$ which we can see are $$A^{[1]}, W^{[2]}$$ and $$b^{[2]}$$. Essentially we now want to find


$$\frac{\partial{Z^{[2]}}}{W^{[2]}}, \frac{\partial{Z^{[2]}}}{b^{[2]}}, \frac{\partial{Z^{[2]}}}{A^{[1]}}.$$

Starting with the first one

$$
\begin{align}
\frac{\partial{Z^{[2]}}}{\partial{W^{[2]}}} &= \\
&= \frac{\partial}{\partial{W^{[2]}}} \cdot \big(W^{[2]}A^{[1]} + b^{[2]}\big) \\
&= A^{[1]T}  \hspace{0.5in} \text{Matrix calculus says this should be transpose}
\end{align}
$$

If you are confused why this should be the transpose I found a great resource explaining this quite well $$\href{https://web.stanford.edu/class/cs224n/readings/gradient-notes.pdf}{here}$$. Let's continue with the next derivative

$$
\begin{align}
\frac{\partial{Z^{[2]}}}{\partial{b^{[2]}}} &= \\
&= \frac{\partial}{\partial{b^{[2]}}} \cdot \big(W^{[2]}A^{[1]} + b^{[2]}\big) \\
&=  \mathbb{I}  \hspace{0.5in} \text{The identity matrix}
\end{align}
$$

The bias term $$b^{[2]}$$ will be a vector of size $$ 1 \times l^{[2]}$$ where $$l^{[2]}$$ are the hidden units in layer 2. One question that might arise when figuring out $$\frac{\partial{Z^{[2]}}}{\partial{b^{[2]}}}$$ and you find out that the derivative is of size $$ n \times l^{[2]}$$, where $$b^{[2]}$$ only is of the size $$ 1 \times l^{[2]}$$. Obviously they need to be equal as you should be able to calculate $$ b^{[2]} - \frac{\partial{Z^{[2]}}}{\partial{b^{[2]}}} $$ and the dimensions should match. The solution when you have multiple incoming gradient for a single node in $$b^{[2]}$$ is to sum them togethor. In the code at the end of this post you will see that when calculating this derivative I will use the sum and the reasoning is as explained here. Now lets find $$\frac{\partial{Z^{[2]}}}{A^{[1]}}.$$

$$
\begin{align}
\frac{\partial{Z^{[2]}}}{\partial{A^{[1]}}} &= \\
&= \frac{\partial}{\partial{A^{[1]}}} \cdot \big(W^{[2]}A^{[1]} + b^{[2]}\big) \\
&=  W^{[2]}  \hspace{0.5in} \text{The identity matrix}
\end{align}
$$

Updating our computational graph it now (a little messier) looks like the following

<div class="imgcap">
<img src="/assets/derivbackprop/backprop2.png" style="border:none; width:90%;">
</div>

Now there is only one tricky part left which I remember was quite confusing. We want to calculate $$\frac{\partial{A^{[1]}}}{Z^{[1]}}.$$ Remember that $$A^{[l]}$$ is simply the rectified linear unit which acts as a gradient router in the way that it only allows nodes greater than 0 to pass. When figuring out this derivative it would be a jacobian matrix but there is a trick one can use here that will save a lot on compute. The ReLU is an elementwise operator, for example if you input a $$ 1 \times 1000$$ vector the output size will be a $$ 1 \times 1000$$. The jacobian matrix formed when doing the derivative of this would be a $$1000 \times 1000$$ dimensional matrix because in theory every value of the first could have influenced every value of the output vector. We know this is not the case as ReLU is an element wise operation. The resulting $$1000 \times 1000$$ dimensional jacobian matrix would therefore actually be a diagonal matrix with 0 and 1's on the diagonal. Let's try to get some intuition for this

$$
\left[\begin{matrix}
    a & b & c \\
\end{matrix}\right]

 \left[\begin{matrix}
    k_{1} & 0  & 0 \\
	0 & k_{2} &  0 \\
    0 &  0   & k_{3}
\end{matrix}\right]
=
\left[
\begin{matrix}
a k_{1} & b k_{2} & c k_{3}
\end{matrix}
\right]
$$

Which we can see would equal the same result if we would have

$$
\left[\begin{matrix}
    a & b & c \\
\end{matrix}\right]
\odot
 \left[\begin{matrix}
    k_{1} &  k_{2} & k_{3} \\
\end{matrix}\right]
=
 \left[\begin{matrix}
	a k_{1} & b k_{2} & c k_{3}\\
\end{matrix}\right]
$$

In other words $$ (1 \times 3)  \times (3 \times 3) = 1 \times 3$$ is the same to $$ (1 \times 3) \odot (1 \times 3) $$ if the $$(3\times 3)$$ matrix is a diagonal matrix. In other words instead of having a $$(100 \times 1000) \times (1000 \times 1000) $$ we can, if the second is a diagonal matrix instead calculate $$(1 \times 1000) \odot (1 \times 1000)$$ which is a lot fewer parameters to keep in memory and calculate.

Now the tricky part is over and we just have to find $$\frac{\partial{Z^{[1]}}}{W^{[1]}}, \frac{\partial{Z^{[1]}}}{b^{[1]}}, \frac{\partial{Z^{[1]}}}{A^{[0]}}.$$ These will be the exact same calculations as the derivatives for $$Z^{[2]}$$ so I will skip them. The last part is to realize when backpropogating and to find the derivatives with respect to let's say $$W^{[2]}$$ you multiply all the arrows connecting $$W^{[2]}$$ to the loss. Essentially this is the chain rule.

$$
\begin{align}
\frac{\partial{Loss}}{\partial{W^{[2]}}} =  \frac{\partial{Loss}}{\partial{Z^{[2]}}} \cdot \frac{\partial{Z^{[2]}}}{\partial{W^{[2]}}}
\end{align}
$$

Which we can see from the computational graph to equal the two backward arrows to $$W^{[2]}$$ times each other. A useful intuition is that the further away you are in the computational graph from the loss, the more chain rules, or arrows times each other would have to be calculated. There is just one thing that is usually used in a neural network which is regularization. It's not particularly hard to calculate the derivatives for but I kept them out of these calculations to make things a little bit cleaner. In the code however I included regularization which I write as lambd.

```python
class NeuralNetwork(object):

    def __init__(self):
        # m for training examples (or how many points)
        self.m = X.shape[0]
        # n for number of features
        self.n = X.shape[1]
        # K for the number of outputs in the last layer (number of classes)
        self.h2 = K
        # h1 for size of first hidden layer
        self.h1 = 25
        #parameter lambd (lambda is keyword in Python) for l2 regularization
        self.lambd = 1e-3
        #learning_rate
        self.learning_rate = 1e-0

    def initalize_he_weights(self, l0, l1):
        # send in previous layer size l0 and next layer size l1 and returns he initialized weights
        w = np.random.randn(l0, l1) * np.sqrt(2. / l0)
        b = np.zeros((1, l1))

        return w, b

    def forward_prop(self, X, parameters):
        W2 = parameters['W2']
        W1 = parameters['W1']
        b2 = parameters['b2']
        b1 = parameters['b1']

        #forward prop
        a0 = X
        z1 = np.dot(a0,W1) + b1
        # apply nonlinearity (relu)
        a1 = np.maximum(0, z1)
        z2 = np.dot(a1, W2) + b2

        #softmax on the last layer
        scores = z2
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        #cache values from forward pass to use for backward pass
        cache = {'a0' : X,
        'probs' : probs,
        'a1' : a1}

        return cache, probs

    def compute_cost(self, y, probs, parameters):
        W2 = parameters['W2']
        W1 = parameters['W1']
        # Want to only take the -np.log of our prediction of the actual class label for
        # each training example. That's why we index [np.arange(self.m), y]
        data_loss = np.sum(-np.log(probs[np.arange(self.m), y]) / self.m)
        reg_loss = 0.5* self.lambd * np.sum(W1*W1) + 0.5*self.lambd*np.sum(W2*W2)

        # total cost J
        J = data_loss + reg_loss

        return J

    def backward_prop(self, cache, parameters, y):
        #Unpack from parameters
        W2 = parameters['W2']
        W1 = parameters['W1']
        b2 = parameters['b2']
        b1 = parameters['b1']

        #Unpack from forward prop
        a0 = cache['a0']
        a1 = cache['a1']
        probs = cache['probs']

        # Start backward propogation
        dz2 = probs
        dz2[np.arange(self.m), y] -= 1
        dz2 /= self.m

        # backprop through values dW2 and db2
        dW2 = np.dot(a1.T, dz2) + self.lambd * W2
#         I = np.identity(300)
#         db2 = np.dot(I, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Back to the (only) hidden layer in this case
        dz1 = np.dot(dz2, W2.T)
        dz1 = dz1 * (a1 > 0)

        #backprop through values dW1, db1
        dW1 = np.dot(a0.T, dz1) + self.lambd * W1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        grads = {'dW1' : dW1,
                 'dW2' : dW2,
                 'db1' : db1,
                 'db2' : db2}

        return grads

    def update_parameters(self, parameters, grads):
        learning_rate = self.learning_rate

        W2 = parameters['W2']
        W1 = parameters['W1']
        b2 = parameters['b2']
        b1 = parameters['b1']

        dW2 = grads['dW2']
        dW1 = grads['dW1']
        db2 = grads['db2']
        db1 = grads['db1']

        W2 -= learning_rate * dW2
        W1 -= learning_rate * dW1

        b2 -= learning_rate * db2
        b1 -= learning_rate * db1

        parameters = {'W1' : W1, 'W2' : W2, 'b1': b1, 'b2': b2}

        return parameters

    def main(self, X, y, num_iter):
        #initialize our weights
        W1, b1 = self.initalize_he_weights(self.n, self.h1)
        W2, b2 = self.initalize_he_weights(self.h1, self.h2)

        #pack parameters into a dictionary
        parameters = {'W1' : W1, 'W2' : W2, 'b1': b1, 'b2': b2}

        # How many gradient descent updates we want to do
        for it in range(num_iter+1):

            # forward prop
            cache, probs = self.forward_prop(X, parameters)

            #calculate cost
            J = self.compute_cost(y, probs, parameters)

            # print cost sometimes
            # notice that we have a cost of about ~1.10 at our first iteration
            # this fits well with the rule of thumb that it should be about -np.ln(1/training_examples)
            # so we are quite confident that our backprop is correct.
            # To be completely sure you would implement gradient checking (which I probably should)
            # but we skip that in this implementation.

            if it % 1000 == 0:
                print(f'At iteration {it} we have a loss of {J}')

            # back prop
            grads = self.backward_prop(cache, parameters, y)

            #update parameters
            parameters = self.update_parameters(parameters, grads)

        return parameters
```
