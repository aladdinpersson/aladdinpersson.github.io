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

> Note: There is a video based tutorial on YouTube which covers the same material as this blogpost,
> and if you prefer to watch rather than read, then you can check out the video [here](https://youtu.be/x9JiIFvlUwk).

In this post we want to get a solid foundation with tensors and their operations which
really lay the foundation to all of deep learning. Linear algebra really. 
All the fancy neural networks, you can imagine all start with tensor operations. 
Explained like that you quickly realize that it's something important that we want to have a deep 
(pun intended) understanding of.

Concretely what we want to understand is the essential tensor operations that we need to know, but 
we will also get a bit beyond those and cover operations for a wide range of potential tasks. 
Before we dive in I think it would be a good idea to just try understand what a tensor actually is. 

In programming terms you can view a tensor like a  multidimensional array, and in more mathematical 
terms a tensor is the generalization of the terms scalar, vector and matrix to higher dimensions. 
A vector is for example a 1 dimensional tensor, and a matrix is a 2 dimensional tensor. Expressed 
generally a tensor is a mathematical structure with shape $(m_1,m_1,m_3, ...)$. 

I will divide this post into a couple of different sections, we will go through:
* **Initialization** methods including type conversions
* **Math** operations on tensors
* **Indexing** in tensors
* **Reshaping** of tensors

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
Let us start with initializing two tensors which we are used to at this point.

```python
x = torch.tensor([1, 2, 3]) 
y = torch.arange([9, 8, 7])
```

And let's start with the absolute simplest and work our way up. How we would add these two
tensors togethor, and with adding I mean elementwise addition so that we obtain 1+9, 2+8 and 3+7.

```python
z = torch.add(x, y) # [10, 10, 10]
z = x + y # [10, 10, 10]
```

Both of the above method are identical, and in many cases there are multiple ways of doing the same thing
in PyTorch. When this is the case I prefer to do the simplest, in this case I always just use +. For subtraction
we can simply do

```python
z = x - y
```

For division we simply do

```python
z = x / y
```

For elementwise multiplication we simply do

```python
z = x * y
```

One additional thing that could be important here is if we would want do make these operations
inplace, meaning we update the original rather than creating a copy which leads to a bit faster
operations. Let's say we wanted to modify x then we could do

```python
x += y # Note, x = x + y is NOT inplace
x -= y # Note, x = x - y is NOT inplace
x /= y # Note, x = x / y is NOT inplace
x *= y # Note, x = x * y is NOT inplace
```

With those basics out of the way let's focus on a little bit more advanced situations and work 
our way up. If we wanted to take the dot product, you could do elementwise multiply togethor
with a summation, but more compactly (and more efficiently) we can just write

```python
z = torch.dot(x, y)
```

If we wanted to do elementwise exponentation we can do this with

```python
z = pow(x, 2)
z = x ** 2
```

and here I prefer the second option because I think it's simpler. Let's take a look at 
how we can also do a couple of quick comparison on our tensors. If we for example wanted to find all elements greater than a certain value, let's
say 1 then we could do this by

```python
z = x > 1 # this will return a binary tensor
```

And this would also work if you instead have a matrix or a higher dimensional tensor. Another useful
comparison operator you can perform if you wanted to check if two tensors are equal (elementwise)
you could do

```python
x = torch.tensor([1,2,3])
y = torch.tensor([1,2,4])
print(x==y) # [True, True, False]
print(torch.eq(x,y)) # [True, True, False]
```

and again here I prefer simply doing ```x == y```.

#### Matrix multiplication
If we start with initalizing two new tensors 

```python
a = torch.randn((2, 5))
b = torch.randn((5, 3))
```

We can then perform matrix multiplication in PyTorch by simply doing:

```python
z = torch.mm(a, b) # mm for matrix multiply
z = a.mm(b)
```

and here both options are equivalent. Perhaps the second is cleaner? I'll leave that for you to 
decide. A bit more general function is `torch.matmul` which works and can work as a replacement
for `torch.mm` but it also works for vectors and will return the dot product in that case. It has
some more advanced features like broadcasting matrix products, but I refer you to the [documentation](https://pytorch.org/docs/stable/generated/torch.matmul.html#torch.matmul) 
for those and we'll take a look at the concept of broadcasting in just a bit.

But first, if we have a matrix that we would like to take matrix multiplied with itself to a large 
power then it could be cumbersome writing many matrix multiplies, then you can instead write

```python
A = torch.randn((5, 5))
power = 10
z = torch.matrix_power(A, power)
``` 

#### Batch Matrix Multiplication
This is a bit more of an advanced operation which is very useful in some scenarios, and at least could
be very good for you to be aware it exists. If we have two three dimensional tensors which I probably first encountered when implementing
Seq2Seq with Attention networks (don't care if you don't know about that) and was confused over how
to do matrix multiplication for a particular dimension when we have three dimensional tensors. So, let's say
we have tensors ´x´ and ´y´ and assume they have shapes ´(B, N, M)` and `(B, M, P)` and we want to
multiply the matrices `(N, M)` with `(M, P)` for every example in our batch `B` without having to
loop through them. This is exactly the use case for `torch.bmm`, so an example is below

```python
b,n,m = 8, 3, 4
b,m,p = 8, 4, 5
x = torch.randn((b, n, m))
y = torch.randn((b, m, p))
z = torch.bmm(x, y) # will be shape (b, n, p)
```

#### Examples of Broadcasting
If you're familiar with numpy then you probably know about broadcasting which is used pretty much
all the time. I think this is easiest explained by an example

```python
x = torch.randn((5, 5))
y = torch.randn((1, 5))
```

Obviously these are not of the same shape, so there is no way we could add them togethor, right? 
Doing this

```python
z = x + y
```

should cause an error. But this is where the magic of broadcasting comes in and what's going to
happen is that the y vector is going to be copied to match the rows of `x` in this case, i.e it will
become a `(5, 5)` matrix with each row consisting of identical values. This works in a similar case
for subtraction etc. 

#### Other useful tensor operations
We can do summation for a tensor with the following

```python
x = torch.randn((5, 5))
z = torch.sum(x, dim=0) # out shape (1, 5)
```

We can obtain the maximum or minimum values of x across a specific dimension by

```python
max_values, max_indices = torch.max(x, dim=0)
min_values, min_indices = torch.min(x, dim=0)
```

Here we also get returned the indices for which the max_values where located, this is also exactly
what the argmax function does, so if you only wanted the indices this is probably the one you should
choose

```python
max_indices = torch.argmax(x, dim=0)
min_indices = torch.argmin(x, dim=0)
```

If we wanted to take the absolute value of each element inside a tensor we could simply do

```python
z = torch.abs(x)
```

If we wanted to obtain the mean across a specific dimension we could do that by

```python
z = torch.mean(x.float(), dim=0)
```

and here I did `x.float()` perhaps unecessarily but what I want to clarify here is that PyTorch
requires the input to be of floating values, for reasons which I am actually not entirely certain.

If you wanted to sort a tensor you can do this by

```python
x = torch.tensor([5,4,3,2,8])
sorted_x, indices = torch.sort(x, dim=0, descending=False)
```

with indices being the order of the indices in order to create the sorted tensor in ascending order.

If we have a tensor passing and we want to only let certain values pass then we can do this with
`torch.clamp` so for example if we wanted to implement ReLU we could do


```python
x = torch.tensor([5,4,-3,2])
x = torch.clamp(x, min=0) # [5, 4, 0, 2]
```

and where you could also specify the max as follows

```python
x = torch.tensor([5,4,3,2,8])
x = torch.clamp(x, min=0, max=4) # [4, 4, 0, 2]
```

Let's say we have a tensor of boolean values and we wanted to do an if statement which evaluates
to True if all the elements inside the tensor are True, we can create this using

```python
x = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
torch.all(x) # will evaluate to False
x = torch.tensor([1, 1, 1, 1], dtype=torch.bool)
torch.all(x) # will evaluate to True
```

on the other hand if you wanted to evaluate to True if any value was True you could do
```python
x = torch.tensor([0, 1, 0], dtype=torch.bool)
torch.any(x) # will evaluate to True
x = torch.tensor([0, 0], dtype=torch.bool)
torch.all(x) # will evaluate to False
```

All right, that was a bunch of tensor operations, and there are many more but these will cover a wide
range of different scenarios.

## Tensor Indexing 
