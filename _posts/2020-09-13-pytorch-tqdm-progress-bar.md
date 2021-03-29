---
layout: post
title: "PyTorch Simple Progress Bar using tqdm"
author: "Aladdin Persson & Sanna Persson"
header-style: text
highlighter: rouge
mathjax: true
math: true
categories: 
  - PyTorch
tags:
  - PyTorch
---


* content
{:toc}


Let us start with the basic imports where we will be using tqdm for our progress bar:

<script src="https://gist.github.com/aladdinpersson/15ba6ec7062551a69a04109fe4fb6443.js"></script>

Let’s create a simple toy dataset example using TensorDataset that we imported above. This is just to make a simple example, so for our dataset we will just generate random numbers. Replace the cell below loading the dataset (doesn’t matter which) of your choice.
<script src="https://gist.github.com/aladdinpersson/06ecd8de8f87401494f335e1fd55eaac.js"></script>

Let’s create a very simple model and training loops:
<script src="https://gist.github.com/aladdinpersson/85bac2c3f8d3801a5d05a39085bd5505.js"></script>

Here we set loss and acc to a random value but here you would set important information you previously computed. This is what it will look like after it’s finished:

Epoch [0/3]: 100%|██████████████████████████████████| 125/125 [00:02<00:00, 42.25it/s, acc=0.776, loss=0.0617]
Epoch [1/3]: 100%|██████████████████████████████████| 125/125 [00:02<00:00, 41.70it/s, acc=0.0216, loss=0.668]
Epoch [2/3]: 100%|██████████████████████████████████| 125/125 [00:03<00:00, 41.23it/s, acc=0.0701, loss=0.912]

Alright so it basically looks identical to how we normally set up our loops in PyTorch. The only difference is that we instead set loop = tqdm(loader) and then we can also add additional information to the progress bar like current (running) accuracy as well as loss for the current batch. Personally I always like to use a progress bar to know how long things will take and I recommend you to do it too! :)