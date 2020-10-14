---
layout: post
title: "PyTorch Neural Network Tutorial"
subtitle: 'How to code a fully connected network in PyTorch'
author: "Aladdin Persson & Sanna Persson"
header-style: text
highlighter: rouge
tags:
  - PyTorch
---
<script data-ad-client="ca-pub-7720049635521188" async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js"></script>

The first step to learning PyTorch is to understand how to structure the code when building a simple neural network.
We will demonstrate this on a fully connected network and train it on the dataset MNIST which contains 28x28
pixels grayscale images of hand-written digits.

<p align="center">

  <img src="../../../../../img/in-post/mnist.png">
  Examples of Mnist images 

</p>

Specifically in this tutorial we will:
* Load the  MNIST dataset from TorchVision datasets
* Build a simple fully connected neural network in PyTorch
* Create a training loop and check accuracy function

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
In PyTorch the general way of building a model is to create a class where the neural network modules you want to use
are defined in the `__init__()` function. These modules can for example be a fully connected layer initialized by 
`nn.Linear(input_features, output_features)`. We then define a function `forward()` in which the forward
propagation of the model is performed by calling the defined modules and applying activation functions from 
`nn.functional` onto the input. When building the model you don't have to think about back-propagation as 
this will be taken care of automatically by PyTorch autograd.
```python
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 
``` 
In this network we use the ReLU activation function and apply it in the forward propagation using ```F.relu()```,
however, PyTorch also allows us to define a module in the ```__init__()``` function for the ReLU activation e.g.
```self.relu = nn.ReLU()``` which can then be reused in the forward propagation each time the activation function is
applied. We should note though that reusing modules defined in the `__init__()` function can only be done for
modules without parameters such as activation functions and pooling layers.

### Setting up the training loop
We have now arrived to the point where we are ready to put it all together and
train the model. First we will define some hyperparameters that are required
for the training of the network. If you are using a GPU for the training and have 
cuda enabled the code will make sure all training is run on the GPU by 
transferring the model and data `.to(device)`.
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 784 # 28x28 is the size of MNIST images
num_classes = 10
learning_rate = 0.001
num_epochs = 1
``` 
Then, we initialize an instance of the model `NN`, the optimizer and the loss function.
When we initialize the model the weights and biases of the model will be initialized
under the hood of PyTorch to random small numbers and if you want a customized weight 
initialization it can be added in the `NN` class. 

In this tutorial we will use the Adam optimizer which is a good default in most applications.
The standard loss function for classifications tasks in PyTorch is the `CrossEntropyLoss()`
which applies the softmax function and negative log likelihood given the predictions
of the model and data labels.
```python
model = NN(input_size=input_size, num_classes=num_classes).to(device)
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

        # Get to correct shape, 28x28->784
        data = data.reshape(data.shape[0], -1) 

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
            x = x.reshape(x.shape[0], -1)

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
If you follow this tutorial you should expect to see a test accuracy of over 90 %
after one epoch of training. 
Leave a comment if you have any thoughts or questions!

Link to [Github](https://github.com/AladdinPerzon)

If you want to watch a video on the content check out: [Neural Network example](https://www.youtube.com/watch?v=Jy4wM2X21u0)
