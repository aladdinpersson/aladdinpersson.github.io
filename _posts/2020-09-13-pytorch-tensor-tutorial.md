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

Note: There is a video based tutorial on YouTube which covers the same
as this blogpost, and if you would prefer to watch rather than read, then you can check out
the video [here](https://youtu.be/x9JiIFvlUwk).


In this post we want to try and get a solid foundation with tensor which
is what all of deep learning is based off. All the fancy neural networks,
you can imagine all start with tensor operations. Explained like that you
quickly realize that it's something important to have a deep (no pun intended)
understanding of.

Concretely what we want to understand is the essential tensor 
operations that we need to know, but we will also get a bit beyond those
and cover operations for a wide range of potential tasks. Before we
dive in I think it would be a good idea to just try understand what a 
tensor actually is. 

In programming terms you can view a tensor like a 
multidimensional array, and in more mathematical terms a tensor is the generalization of 
the terms scalar, vector and matrix to higher
dimensions. Here a vector is a 1 dimensional tensor, and a matrix is a 
2 dimensional tensor. Expressed generally a tensor is a mathematical structure with shape 
$(m_1,m_1,m_3, ...)$ is called a tensor. 

I will divide this post into a couple of different sections, we will go through:
* Tensor initialization methods
* Math operations on tensors
* Indexing in tensors
* Reshaping of tensors

To start off we can define the device that the tensors should run on 
which we primarily set to the GPU if you have one enabled, otherwise
we just set it to the cpu. 

``` python
device = "cuda" if torch.cuda.is_available() else "cpu"
```
### Initialization
Let's start simple, let's say we want to initialize a tensor manually
with the two rows containing the values [1, 2, 3] and [4, 5, 6]. We can
easily do this by writing this inside a nested list and then convert by
calling `torch.tensor()`
 
```python
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
```

We can also add a couple of arguments to specify the details of how we want to initialize this
tensor

```python
my_tensor = torch.tensor(
    [[1, 2, 3], [4, 5, 6]], 
    dtype=torch.float32, 
    device=device, 
    requires_grad=True
)
```

Here the argument `dtype` specifies what type you want to store the entries in the tensor 
in, which in our case we want to be set the 32 bit floating point. The device is simply what we
initialized at the top which means the tensors will be stored on VRAM and run on the GPU as our
primary choice. The `requires_grad` determines whether you would like PyTorch should calculate
a gradient with respect to the tensor. This argument is important to set to true if the tensor is
part of a model you want to perform back-propagation on, but is not something we will go into 
deep depth on in this tutorial. 

If we encounter a tensor we can also obtain these properties of the tensor by printing them.
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

print(f"Shape of tensor {my_tensor.shape}")  # in this case 2x3
print(f"Requires gradient: {my_tensor.requires_grad}")  # true/false
```

There are also many other ways of initializing tensor with other properties. Let us take a look
at a couple of the important ones. The function `torch.empty` returns a tensor of the 
specified shape in `size` with uninitialized data. I've noticed that this function can sometimes
be confusing, note that the data here is not intialized as zeros, but can be "random" values
consisting of things that were in your computer buffer.

```python
size=(3, 3)
x = torch.empty(size=size)  # Tensor of shape 3x3
```

The function `torch.zeros` returns a tensor of the specified shape or size that is filled with zeros.

```python
x = torch.zeros((3, 3))  # Tensor of shape 3x3 
```

The function `torch.ones` returns a tensor of the specified shape that is filled with ones.

```python
x = torch.ones((3, 3))  # Tensor of shape 3x3 
```

The function `torch.eye` (**I** see what you did there) returns an identity
matrix of the specified shape. Note that here the shape should not be a tuple, as it is only
a 2 dimensional tensor.

```python
x = torch.eye(5, 5)  # Tensor of shape 5x5
```

The `torch.arange` function returns a vector with integer entries ordered from `start` to
non-including the `end` 
where you can also specify the`step`, this is very similar to the python `range` function. 

```python
x = torch.arange(start=0, end=5, step=1)
```

The `torch.linspace` function returns a vector of length `steps` with equally 
spaced intervals of numbers ranging from `start` to `end`.

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

The functions `torch.diag` returns a matrix with the same shape as the 
input with only the diagonal entries left.

```python
x = torch.diag(torch.ones(3))  # returns 3x3 identity matrix
```

#### Conversion between types
Let's initialize a tensor

```python
tensor = torch.arange(4)  # [0, 1, 2, 3] 
```

which by default will be initialized as int64. We will then show how we can 
convert the tensor to other types. The ones that you will encounter most often are:

* Conversion to float32 by `tensor.float()`
* Convert to float16 by `tensor.half()`
* Conversion back to int64 by `tensor.long()`

Note, however, that the conversions are not inplace and if you want to 
change the type you have to do e.g. `tensor = tensor.float()`. For other types you can
also do the following:

* Convert to int16 by `tensor.short()`
* Conversion to float64 by `tensor.double()`
* Convert to boolean i.e. 1 if non-zero entry else 0 by `tensor.bool()`  

Different functions in PyTorch will accept different types and it is likely you will
face many errors due to wrong input types before you are used to them. 

### Mathematical operations
hello