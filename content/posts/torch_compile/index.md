---
title: "Torch compile"
date: 2024-09-01
draft: false
description: ""
slug: "torch-compile"
tags: ["torch-compile", "compiler"]
---
Remember when machine learning was done using Caffe? The ML Surgeon remembers that.
If you didn’t catch the reference, too bad for you!

In the last few days, I've been reflecting on how much easier machine learning has become in recent years. Not only do we now have a larger and higher-quality plethora of tools and frameworks—ones we could only dream of a few years ago—but the fact that these frameworks are so user-friendly is mind-boggling!

This got me thinking: are practitioners truly aware of the extreme complexity behind modern machine learning tools? Probably not. That’s why today, I want to dissect **Torch Compile**.

This article will be quite complex and lengthy, so gear up. But first, sterilize your hands.


## Dissecting torch.compile: A Surgeon’s Approach to PyTorch Optimization
At the end of 2022, PyTorch 2.0 was released, bringing with it a host of improvements and new features. Among them, the standout addition was undoubtedly **torch.compile**, a method designed to speed up PyTorch code. Its usage is quite straightforward: pass either a `torch.nn.Module` or a function to the method, and you’ll get an optimized version of it. For example:

```python3
class MyModel(torch.nn.Module)
    ...

model = MyModel()
optimized_model = torch.compile(model)
```

The `optimized_model` will, hopefully, run faster than the originally instantiated model. Later on, we’ll conduct some benchmarks to demonstrate these speedups. So far, so simple, right?

But what actually happens when we use `torch.compile`? How can a single line of code optimize a model and achieve up to 3x speedups compared to classic, eager-mode PyTorch?

To understand that, we’ll need to cut deeper—time to get some blood on our hands (or, ideally, gloves).


## Things to know before cutting deep
Before we dive into the complexities hidden within `torch.compile`, it’s essential to cover some foundational concepts. These basics are crucial for understanding the roles of the tools involved in the compilation pipeline. Trust me, the intricacies behind `torch.compile` are quite convoluted, and there’s a lot to grasp before you can see the full picture. So, please be patient and make sure you fully understand each step before proceeding to the next.

### Slicing Through the Layers: Dissecting PyTorch’s Computational Graphs
Have you ever wondered what really happens when you execute PyTorch code? If not, shame on you! Don’t take for granted the incredible features right at your fingertips!

Let’s start from the beginning. Suppose you have code like this pseudocode:

```
fn(x op y)
```

where, `x` and `y` are two tensors, `op` is an operation (like the product `*` or sum `+`), and `fn` is a function (like `log` or `sin`).

This syntax results in a **computational graph**, which represents the operations that will be performed on the tensors. Below is a simple sketch of the computational graph that would result from the code above:

![Forward Computational Graph](forward_computational_graph.png "Is this crooked? I can't really tell.")

In Pytorch, the graph is built **dynamically** as operations are applied to tensors, which is often referred as *define-by-run*.

But wait! That's only the forward step! We can’t train our models with just that!
Luckily for us, Pytorch provides [**autograd**](https://pytorch.org/docs/stable/autograd.html), a system responsible for automatic differentiation. It records operations on tensors to form an **autograd graph**.
Long story short, PyTorch automatically computes gradients for tensors. A computational graph for the backward pass looks something like this:

![Backward Computational Graph](backward_computational_graph.png "It's definitely crooked")

Damn, I'm good at drawing. Anyway, in case you missed your calculus classes, the notation {{< katex >}}\\(\frac{\partial Z}{\partial X}\\) and {{< katex >}}\\(\frac{\partial Z}{\partial Y}\\) stands for the partial derivative of `Z` with respect to `X` (or `Y`).

As you can see, there are a lot of graphs involved. So, guess what `torch.compile` does to these graphs? That’s right—it optimizes them to make the overall computation faster.

### Probing the Depths: Surgical Insights into PyTorch’s FX Graphs
Let’s dive deeper into the world of graphs. In the previous section, we explored the critical role of computational graphs. But how do we go about optimizing them? I’m not referring to techniques or methodologies—I'm talking about the nuts and bolts of how we can technically modify and optimize PyTorch's computational graphs.

To do that, we need a specialized toolkit. Fortunately, PyTorch equips us with just what we need: the [**FX**](https://arxiv.org/abs/2112.08429) toolkit.

The FX toolkit allows to modify `torch.nn.Module`s by implementing a pipeline consisting of a **symbolic tracer**, an **intermediate representation** (IR) and a **Python code generator**. This makes FX a powerful Python-to-Python transformation toolkit.

The symbolic tracer constructs a `torch.fx.GraphModule` by recording the operations that occur when the `nn.Module` is fed with fake data, called **proxies**. 

A `GraphModule` is essentially a `nn.Module` generated from a `torch.fx.Graph`, which serves as the core data structure for FX’s internal representation.

With just a few lines of code, we can observe how the symbolic tracer and intermediate representation function:

```python
import torch
import torch.fx
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.rand(4, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        return (self.linear(x) + x).relu()

m = MyModule()
gm = torch.fx.symbolic_trace(m)
print(gm.graph)
```
This script outputs the following graph representation:

```
graph():
    %x : [num_users=2] = placeholder[target=x]
    %linear : [num_users=1] = call_module[target=linear](args = (%x,), kwargs = {})
    %add : [num_users=1] = call_function[target=operator.add](args = (%linear, %x), kwargs = {})
    %relu : [num_users=1] = call_method[target=relu](args = (%add,), kwargs = {})
    return relu
```

This output shows the graph’s intermediate representation, which is made up of `Node`s. Without going too deep into the details, you can see references to module calls (e.g. `linear`), function calls (e.g. `add`), and method calls (e.g. `relu`). Each node also specifies the **args** for the operation, which are other nodes within the graph.

Once we have this graph, we can modify it as needed. Afterward, the **code generator** component takes over, creating a new `GraphModule` from the modified `Graph` data structure. I won’t dive into the specific techniques for modifying a graph here—this article is already long enough!


### Stitching Together Efficiency: Introducing CUDA Graphs
While we're on the subject of graphs, it’s worth highlighting another important feature: **CUDA Graphs**. Introduced in 2021, CUDA Graphs are a relatively new addition to the PyTorch ecosystem, specifically available for NVIDIA GPUs with CUDA version 10 or higher.

Typically, when operations are executed on the GPU, each kernel launch must be initiated from the CPU—a process that introduces noticeable overhead, especially when dealing with thousands of operations. Each individual launch might be small, but when accumulated, this overhead can impact performance.

CUDA Graphs address this by representing GPU operations as a single, cohesive graph. While building and launching this graph may initially be slower, the advantage lies in the fact that all subsequent operations remain on the GPU, significantly reducing the overhead caused by CPU-GPU communication.

The image below illustrates this concept perfectly:

![](cuda_graph.png "[Credits to this blogpost](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)")


## Into the Operating Room: Dissecting the Mechanics of Torch Compile
After all this talk about graphs, it's finally time to get down to business. Now, we’re ready to make the incision and dive deep into `torch.compile` to explore its inner workings. Armed with the knowledge we’ve gained in the previous sections, this should feel like a well-prepared field trip into the body of PyTorch, right? I certainly hope so—my head’s already spinning from the sheer complexity of it all!

### Torch Dynamo


## Diocan
Hey! Torch just called me GPU-poor! Not cool! 
https://discuss.pytorch.org/t/torch-compile-warning-not-enough-sms-to-use-max-autotune-gemm-mode/184405
https://github.com/pytorch/pytorch/blob/e5f5bcf6d4ec022558caf4d0611d928497394a88/torch/_inductor/utils.py#L644-L645
