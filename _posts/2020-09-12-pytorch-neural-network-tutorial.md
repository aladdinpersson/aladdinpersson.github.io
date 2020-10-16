***
Note: There is a video based tutorial on YouTube which covers the same material as this blogpost, and if you prefer to watch rather than read, then you can check out the video [here](https://www.youtube.com/watch?v=Jy4wM2X21u0).
***

In this post we will learn how to build a simple neural network in PyTorch and also how to train it
to classify images of handwritten digits in a very common dataset called MNIST.

Specifically in this tutorial we will:
* Load the  MNIST dataset
* Build a simple fully connected neural network in PyTorch
* See how to train this network
* Check the accuracy of our network on training data and testing data

### Imports
First we need will need a couple of different packages


https://gist.github.com/58fddaa31ca840acec525e05de0da741

For loading the classical dataset [MNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#mnist) we need the following packages
from PyTorch we can do this using torchvision as follows


https://gist.github.com/006db2cd57571d14ad786340084840ec

Note that `torchvision.datasets` contains many more "standard datasets" that you may want to play
around with as well,
such as [CIFAR-10](https://pytorch.org/docs/stable/torchvision/datasets.html#cifar) and [SVHN](https://pytorch.org/docs/stable/torchvision/datasets.html#svhn) which can easily be loaded into PyTorch.  

### Loading the data
In this tutorial we use `torchvision.datasets` to load the data and if you are starting out learning deep
learning they provide several datasets you can start working with before you dive into 
custom datasets. The following lines are all that's needed to load the MNIST train and test data. Here we are also setting a `batch size` which will be the amount of examples our network will see at a time when performing update steps.


https://gist.github.com/02e6e466c1a693c45ab5d85d93f15e34

The `train_dataset` and `test_dataset` are `Torchvision` dataset objects and in this example the only transform we apply to the images and labels is to convert them to PyTorch tensors with `transforms.ToTensor()` and this is a necessary step to train our network. The `DataLoader()` returns an iterator which will generate batches of the selected batch_size as tuples of `(data, labels)`. We will therefor obtain 64 images at the same time from a batch with the associated correct label digit with those images. The argument `shuffle` determines whether these batches will be shuffled and you can default to setting it to True unless you are working with inherent sequential data. New batches will then be randomly selected each epoch which makes sure that the 64 examples inside a batch will be different from epoch to epoch.

### Building the model
In PyTorch the general way of building a model is to create a class where the neural network modules you want to use
are defined in the `__init__()` function. These modules can for example be a fully connected layer initialized by 
`nn.Linear(input_features, output_features)`. We then define a function `forward()` in which the forward
propagation of the model is performed by calling the defined modules and applying activation functions from `nn.functional` onto the input. When building the model you don't have to think about back-propagation as 
this will be taken care of automatically by PyTorch autograd.


https://gist.github.com/f4964bf51bcedce288defae4652ae4fc

In this network we use the rectified nonlinear unit (ReLU) activation function and apply it in the forward propagation using ```F.relu()``` and the fully connected network has one input layer, one hidden layer and one output layer with 10 nodes, one for each digit 0-9.

### Setting up the training loop
We have now arrived to the point where we are ready to put it all together and
train the model. First we will define some hyperparameters that are required
for the training of the network. If you are using a GPU for the training and have 
cuda enabled the code will make sure all training is run on the GPU by 
transferring the model and data `.to(device)`.



https://gist.github.com/f5e14ec24d3b04da628b3356a7c878e2

Then, we initialize an instance of the model `NN`, the optimizer and the loss function.
When we initialize the model the weights and biases of the model will be initialized
under the hood of PyTorch to random small numbers and if you want a customized weight 
initialization it can be added in the `NN` class. 

In this tutorial we will use the Adam optimizer which is a good default in most applications.
The standard loss function for classifications tasks in PyTorch is the `CrossEntropyLoss()`
which applies the softmax function and negative log likelihood given the predictions
of the model and data labels. This is also the reason why we do not apply softmax to the outputs from our neural network, because it is already included in `CrossEntropyLoss` and we do not want to apply it twice.


https://gist.github.com/53f5890be03ba3bade02d432ad878865

Now, we are ready to start training the model. We will loop through the epochs
and then the train loader. For each batch we will perform forward propagation, 
compute the loss, calculate the gradients in back-propagation and update the weights
with the optimizer. 



https://gist.github.com/6039b83bb56aea5dfe9c2b7044a5fb24

    Epoch: 0
    Epoch: 1
    Epoch: 2
    

Lastly, we want to make sure our model is learning by computing the accuracy on
the training and test set. We will therefore code a function which checks the
accuracy given the model and dataloader. In PyTorch a model can either be in the
`train` or `eval` state which determines the behaviour of certain modules such as dropout
or batch normalization that should act differently during training and testing time (don't bother too much if you don't know what those are). 
The default mode is `train` and when you evaluate a model you should toggle the model to
`eval`. If you then want to continue training you would have to toggle it back to `train`.
The attentive reader will realize that in this example, however, no module has different
behaviour on train and test time. During evaluation you do not either want to spend
valuable time on computing the gradients and all computations should therefore be
under `with torch.no_grad()`. To check the accuracy of the model we will perform 
a forward pass on all batches in the dataloader, compute the predictions and then 
count how many predictions that correspond to the correct labels. 



<script src="https://gist.github.com/7844a811f7659783f9763eb6870c6642.js"> </script>

<script src="https://gist.github.com/aladdinpersson/808836ea39d790e1e6bdea2cb0573ea2.js"> </script>

    Got 58613 / 60000 with accuracy 97.69
    Got 9687 / 10000 with accuracy 96.87
    

If you follow this tutorial you should expect to see a test accuracy of over 95% after three epochs of training. So after following this tutorial you learned how to setup a neural network in PyTorch, how to load data, train the network and finally see how well it performs on training and test data! 

Leave a comment preferably on the [YouTube video](https://www.youtube.com/watch?v=Jy4wM2X21u0) if you have any thoughts or questions! [Here](https://github.com/AladdinPersson/machine-learning-collection) is a link to the Github repository where you can find this code, and many more similar to this!