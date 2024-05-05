---
title: "An Introduction to Sparsity for Efficient Neural Network Inference"
date: 2024-05-05
draft: false
description: "Sparsity is a solution to reduce the number of parameters and number of operations in Neural Networks, granting outstanding computational speedups and memory savings during inference."
slug: "advanced-customisation"
tags: ["advanced", "css", "docs"]
---

**Large Language Model**. How many times did you read that term? Nowadays, the popularity of Artificial Intelligence is to be attributed to the exceptional results obtained, in the past few years, by applications that leverage large models. Surely, you know the most popular one, ChatGPT, made by OpenAI.

When I first started learning about neural networks -about 5 years ago-, one of the key questions I had was if it would be possible to know, a priori, the minimum number of parameters that a network must implement to achieve a certain metric value when solving a specific problem. Unfortunately, as far as I know, there is no such theorem. Surely, we know about convergence theorems -like the [Universal Approximation Theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem)-, but nothing tells us the optimal number of parameters, given an architecture and a problem to solve.

A very interesting methodology is the Cascade Correlation, which is a constructive approach for obtaining a network that solves a specific problem. However, this methodology is impractical in modern times: the size of neural networks is just too big, especially when talking about LLMs, which typically span from 7 to 70 billion parameters.

Therefore, a follow-up question: given a trained, large network, can we reduce its size while maintaining the same accuracy?

That’s the key idea behind Pruning.


## Pruning for Sparsity
Let’s start from a biological point of view: humans drastically reduce the number of synapses per neuron from early childhood to adolescence. This has been shown in various papers (e.g. [Synapse elimination accompanies functional plasticity in hippocampal neurons](https://pubmed.ncbi.nlm.nih.gov/18287055/)) and it is a way for the brain to optimize its knowledge and remove redundancy. As often happens, this concept can also be applied to learning architectures.

Assume to have a relatively large neural network that has been trained to approximate the solving function for a given problem, which was described by a dataset. This means that, by training, we minimized the loss function defined as:

{{< katex >}}
\\(\mathcal{L}(\theta, \mathcal{D}) = \frac{1}{N}\sum_{(x,y)\in(X,Y)}{L(\theta, x, y)}\\)

where \\(\theta\\) represents the weights of the network and \\(\mathcal{D}\\) symbolizes the data used to train it.

Let’s now assume that we are capable of pruning the weights \\(\theta\\), therefore obtaining \\(\theta_{pruned}\\). Our objective is to get approximately the same loss value when using the pruned weights. In math:

The advantage here is obvious: \\(\theta_{pruned}\\) is a smaller, more efficient model that is capable of replicating the accuracy of a bigger, slower model.

Take a look at the picture below: by pruning the red neuron, we remove 8 connections and the activation of the neuron itself. If both the networks had the same accuracy, we would prefer the right one, because its inference is more efficient.

![Nvidia Blog](nvidia_blog.webp "credits to [Nvidia blog](https://developer.nvidia.com/blog/wp-content/uploads/2019/03/remove_neuron.png)")
