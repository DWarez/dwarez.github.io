---
title: "Quantization"
date: 2024-11-10
draft: false
description: ""
slug: "quantization"
tags: ["quantization", "inference", "optimization"]
---

As humans, we perceive the space and time around us as continuous. This continuity suggests that the concept of infinity is intrinsic to nature. However, some theories challenge this view, proposing that reality might, in fact, be discrete. For instance, **string theory** posits that the fundamental building blocks of matter are tiny, vibrating strings, with their vibrations determining the properties of matter. Similarly, **loop quantum gravity** suggests that space itself is composed of "space grains," effectively making the fabric of the universe discrete.

Why does this matter to us, the Machine Learning Surgeons? Because we work with machines—and machines operate in a discrete world, no exceptions. This raises a critical question: If everything we achieve with machines is built on finite data, how do we define the data types we use? Moreover, what are the implications of choosing one data type over another when running machine learning models?

Let’s dissect the topic and dive into quantization!


## What is quantization
In the natural world, many signals—such as sound waves, light intensity, or even time itself—are continuous. However, machines, by their very nature, operate in a discrete framework. They deal with bits, bytes, and finite representations of data. This fundamental limitation means that to store and process real-world signals, we must first translate them into a form that machines can understand. Enter **quantization**.

Quantization is the process of mapping a continuous range of values into a finite set of discrete levels. Think of it as breaking down a flowing river of data into buckets that can be cataloged and stored. For example:

* An audio signal, with its infinite variations in amplitude, must be sampled at specific intervals and its amplitude mapped to discrete levels.
* An image, representing continuous changes in light and color, is pixelated into finite, numerical values.

This conversion is essential for computation but comes with trade-offs. Quantization introduces **approximations**; a continuous signal can only be represented with a finite precision, leading to quantization errors. In fact, measure theory states that infinite precision is not achievable, not even in a theoretical setting. 

The following image illustrates the process of quantization. First, the signal is sampled at a specific rate, meaning values are selected along the x-axis at regular intervals. Here, we’ll assume the x-axis represents time. On the y-axis, we have the amplitude of the signal, which is then mapped to a finite set of discrete levels.

The difference between the actual signal value and its closest quantized level is known as the **quantization error**. This error is an unavoidable artifact of the process, stemming from the approximation required to fit continuous values into a discrete framework:

![sin](sin.png "Credits to [HanLab](https://hanlab.mit.edu)")

At first glance, you might think this discussion has little to do with machine learning, especially since we’re not directly talking about models. Why should we care about the quantization of real-valued, continuous signals like audio or images? And you’d be partly correct—our primary concern isn’t the raw quantization of these signals.

Instead, the point here is to emphasize the **tradeoff** between the true value of a signal and its **representation within hardware**. Neural networks aim to mimic the inner workings of the human brain, which constantly produces electrical impulses and signals. In machine learning, these impulses are represented as **neuron activations**. However, these activations must ultimately exist within a machine, and machines operate within the constraints of discreteness and finiteness. This means that the continuous signals we’re trying to emulate must be mapped into a discrete, finite set of values.

This is where data types come into play. The choice of data types for representing weights and activations in neural networks is absolutely critical. It impacts not only the **precision** and **accuracy** of the computations but also the efficiency of the entire system. And, as you’ll see shortly, the requirements for data representation often differ significantly between the training and inference phases of a model.

Before diving into those differences, let’s take a moment to refresh our understanding of numeric data types and their implications.