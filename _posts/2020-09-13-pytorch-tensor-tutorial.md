---
layout: post
title: "Complete PyTorch Tensor Tutorial"
subtitle: 'The basics and beyond of Tensor operations'
author: "Aladdin Persson & Sanna Persson"
header-style: text
highlighter: rouge
mathjax: true
math: true
tags:
  - PyTorch
---
<script data-ad-client="ca-pub-7720049635521188" async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js"></script>

A tensor is generalization of the terms scalar, vector and matrix to higher
dimensions for example a mathematical structure with shape $(m_1,m_1,m_3, ...)$ is called
a tensor. Deep learning methods are all based on manipulating tensors by combining
relatively simple operations into complex architecture. We will in this tutorial go
through the most important tensor operations that you will need for a while. 

Specifically, we will go through:
* Tensor initialization methods
* Math operations on tensors
* Indexing in tensors
* Reshaping of tensors

First, just define the device which will be the GPU if you have
CUDA enabled with PyTorch. 
``` python
device = "cuda" if torch.cuda.is_available() else "cpu"
```
### Initialization
Starting from the basics with different methods of initializing tensors. If you
have a few values in nested lists that you want to insert in a tensor by hand we can just
convert them by `torch.tensor()`. 
```python
my_tensor = torch.tensor(
    [[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device, requires_grad=True
)
```
The argument `dtype` specifies what type you want to store the entries in the tensor 
in and `requires_grad` determines whether you would like PyTorch should calculate
a gradient with respect to the tensor. This argument is important to set to true if the tensor is
part of a model you want to perform back-propagation on. 

If we encounter a tensor we can also obtain these properties of the tensor by printinng them.
```python
print(
    f"Information about tensor: {my_tensor}"
)  # Prints data of the tensor, device and grad info

print(
    "Type of Tensor {my_tensor.dtype}"
)  # Prints dtype of the tensor (torch.float32, etc)

print(
    f"Device Tensor is on {my_tensor.device}"
)  # Prints cpu/cuda (followed by gpu number)

print(f"Shape of tensor {my_tensor.shape}")  # Prints shape, in this case 2x3

print(f"Requires gradient: {my_tensor.requires_grad}")  # Prints true/false
```
There are also many other ways of initializing tensor with other properties and
we will mention the ones that you probably will encounter most often.
The function `torch.empty` returns a tensor of the specified shape in `size` with uninitialized data.
```python
x = torch.empty(size=(3, 3))  # Tensor of shape 3x3
``` 
The function `torch.zeros` returns a tensor of the specified shape that is filled with zeros.
```python
x = torch.zeros((3, 3))  # Tensor of shape 3x3 
```  
The function `torch.ones` returns a tensor of the specified shape that is filled with ones.
```python
x = torch.ones((3, 3))  # Tensor of shape 3x3 
```
The function `torch.eye` (**I** see what you did there) returns an identity matrix of the specified shape.  
```python
x = torch.eye(5, 5)  # Tensor of shape 5x5
```
The `torch.arange` function returns a vector with integer entries ordered from `start` to `end` with `step` as interval 
```python
x = torch.arange(start=0, end=5, step=1)
```
The `torch.linspace` function returns a vector of length `steps` with equally spaced intervals of numbers ranging from `start`
to `end`.
```python
x = torch.linspace(start=0.1, end=1, steps=10)  
```
The following will yield a tensor of the specified shape with entries drawn from the
normal distribution with the mean and standard deviation given as arguments.
```python
x = torch.empty(size=(1, 5)).normal_(mean=0, std=1)
```
If we instead want to have values drawn from a uniform distribution in a specified
range we can do:
```python
x = torch.empty(size=(1, 5)).uniform_(0, 1) # range (0,1)
```
The functions `torch.diag` returns a matrix with the same shape as the input with only the diagonal entries left.
```python
x = torch.diag(torch.ones(3))  # returns 3x3 identity matrix
```

### Conversion between types
Let's initialize a tensor
```python
tensor = torch.arange(4)  # [0, 1, 2, 3] 
```
which by default will be initialized as int64. We will then show how we can 
convert the tensor to other types. 
The ones that you will encounter most often are:

* Conversion to float32 by `tensor.float()`
* Conversion to float64 by `tensor.double()`
* Conversion back to int64 by `tensor.long()`

Note, however, that the conversions are not inplace and if you want to 
change the type you have to do e.g. `tensor = tensor.float()`. Less commonly you 
will see the following conversions in code. 

* Convert to int16 by `tensor.short()`
* Convert to float16 by `tensor.half()`
* Convert to boolean i.e. 1 if non-zero entry else 0 by `tensor.bool()`  

Different functions in PyTorch will accept different types and it is likely you will
face many errors due to wrong input types before you are used to them. 

### Mathematical operations

Link to [Github](https://github.com/AladdinPerzon)

If you want to watch a video on the content check out: [Complete Pytorch Tensor Tutorial](https://youtu.be/x9JiIFvlUwk)
