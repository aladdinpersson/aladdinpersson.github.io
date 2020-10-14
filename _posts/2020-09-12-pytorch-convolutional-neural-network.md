---
layout: post
title: "PyTorch Convolutional Neural Network Tutorial"
subtitle: 'How to code a Convolutional Neural network in PyTorch'
author: "Aladdin Persson & Sanna Persson"
header-style: text
highlighter: rouge
mathjax: true
math: true
tags:
  - PyTorch
---	
<script data-ad-client="ca-pub-7720049635521188" async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js"></script>

The state of the art models in image recognition are all built with convolutional
neural networks. Today many even claim that the problem of computer vision is solved
as deep learning models outperform humans on many vision tasks. Convolutional neural
networks is therefore one of the central pillar of deep learning. In this tutorial 
we will code a basic convolutional neural network and train it on the dataset MNIST which contains 28x28
pixels grayscale images of hand-written digits.

<p align="center">

  <img src="../../../../../img/in-post/mnist.png">
  Examples of Mnist images 

</p>

Specifically in this tutorial we will:
* Load the  MNIST dataset from TorchVision datasets
* Build a simple convolutional neural network in PyTorch
* Create a training loop and check accuracy function

If you have already read the tutorial on building [fully connected neural network](2020-09-12-pytorch-convolutional-neural-network.md)
you will find that the parts imports, loading the data and setting up the training loop
are almost left unchanged. This tutorial assumes that you know the basics of coding
a neural network in PyTorch, otherwise check out the tutorial mentioned above.  
### Imports
First, we need to import the packages required for the code. To create 
models in PyTorch the following packages are required. 
```python
import torch # just torch
import torchvision # popular datasets, models etc.
import torch.nn as nn # all neural networks modules
import torch.optim as optim # all optimizers 
import torch.nn.functional as F # functions without parameters 
```
For loading the classical dataset [MNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#mnist) we need the following packages
from PyTorch. 
```python
from torch.utils.data import DataLoader # data management 
import torchvision.datasets as datasets # standard datasets
import torchvision.transforms as transforms # data processing
```  
Note that `torchvision.datasets` contains many standard datasets
such as [CIFAR-10](https://pytorch.org/docs/stable/torchvision/datasets.html#cifar) and [SVHN](https://pytorch.org/docs/stable/torchvision/datasets.html#svhn) which can easily be loaded into PyTorch.  
### Loading the data
In this tutorial we use `torchvision.datasets` to load the data and if you are starting out learning deep
learning they provide several datasets you can start working with to learn modelling and training before you dive into 
custom datasets. The following lines are all that's needed to load the MNIST train and test data.

```python
batch_size = 64
train_dataset = datasets.MNIST( 
    root="dataset/",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)
test_dataset = datasets.MNIST(
    root="dataset/",
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=True
)
``` 
The `train_dataset` and `test_dataset` are Torchvision dataset objects and in this example
the only transform we apply to the images and labels is to convert them to PyTorch tensors with
`transforms.ToTensor()`. The `DataLoader()` returns an iterator which will generate batches of the selected 
`batch_size` as tuples of (data, labels). The argument `shuffle` determines
whether these batches will be shuffled and you can default to setting it to true
if your specific use case does not suggest otherwise. New batches will then be randomly selected each epoch
which makes sure the batches are representative of the data and that the gradients
in each epoch are different.

### Building the model
A convolutional network for image classification is typically structured in the way that you have a number 
of blocks with convolutional layers and then a few fully connected layers before the classification is made.
In each block there is convolutional layer followed by, often batch normalization and 
the ReLU activation function. Between some of the blocks a maxpooling layer is used to
scale down the feature maps.
There are many famous variations of this structure (VGG and ResNet variations, GoogLeNet etc.) and we will only make a simple example
following this recipe. We will use two convolutional layers with the ReLU activation function
and apply a maxpooling layer after each convolutional layer. We will then reshape the feature maps and
pass them through a single fully connected layer to perform the classification.
```python
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x
``` 
In each convolutional layer the `kernel_size`, the number of `out_channels`, the `stride`
and the `padding` are model choices which will affect the model's performance. To gain
insight into how we can choose these we can calculate the output size from each 
convolutional layer with the following formula

$\text{output shape} = \frac{\text{input shape} - \text{kernel size} + 2\times \text{padding}}{\text{stride}} +1$.

The output shape for a max pooling layer can also be computed in the same way but
you will often see in code that the max pooling layer will have kernel size 2x2 and stride
of 2x2 which will just halve the input in the image dimension. To simplify the model design
it is common to use padding such that the output shape and input shape are equal in
the convolutional layer and then halve the size of the image features with a maxpooling layer when
you want to scale them down. This is the method used in our simple convolutional network
above. 

### Setting up the training loop
We have now arrived to the point where we are ready to put it all together and
train the model. First we will define some hyperparameters that are required
for the training of the network. If you are using a GPU for the training and have 
cuda enabled the code will make sure all training is run on the GPU by 
transferring the model and data `.to(device)`.
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_channel = 1
num_classes = 10
learning_rate = 0.001
num_epochs = 5
``` 
Then, we initialize an instance of the model `CNN`, the optimizer and the loss function.
When we initialize the model the weights and biases of the model will be initialized
under the hood of PyTorch to random small numbers and if you want a customized weight 
initialization it can be added in the `CNN` class. 

In this tutorial we will use the Adam optimizer which is a good default in most applications.
The standard loss function for classifications tasks in PyTorch is the `CrossEntropyLoss()`
which applies the softmax function and negative log likelihood given the predictions
of the model and data labels.
```python
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```
Now, we are ready to start training the model. We will loop through the epochs
and then the train loader. For each batch we will perform forward propagation, 
compute the loss, calculate the gradients in back-propagation and update the weights
with the optimizer. 

```python
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward propagation
        scores = model(data)
        loss = criterion(scores, targets)

        # zero previous gradients
        optimizer.zero_grad()
        # back-propagation
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
```
Lastly, we want to make sure our model is learning by computing the accuracy on
the training and test set. We will therefore code a function which checks the
accuracy given the model and dataloader. In PyTorch a model can either be in the
`train` or `eval` state which determines the behaviour of modules such as dropout
or batch normalization that should act differently during training and testing time. 
The default mode is `train` and when you evaluate a model you should toggle the model to
`eval`. If you then want to continue training you would have to toggle it back to `train`.
The attentive reader will realize that in this example, however, no module has different
behaviour on train and test time. During evaluation you do not either want to spend
valuable time on computing the gradients and all computations should therefore be
under `with torch.no_grad()`. To check the accuracy of the model we will perform 
a forward pass on all batches in the dataloader, compute the predictions and then 
count how many predictions that correspond to the correct labels. 
```python
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy"
            f" {float(num_correct) / float(num_samples) * 100:.2f}"
        )

    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
```
If you follow this tutorial you should expect to see a test accuracy of over 98 %
after five epochs of training. 
Leave a comment if you have any thoughts or questions!

Link to [Github](https://github.com/AladdinPerzon)

If you want to watch a video on the content check out: [Pytorch CNN example](https://www.youtube.com/watch?v=wnK3uWv_WkU)
