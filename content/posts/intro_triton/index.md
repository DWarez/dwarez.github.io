---
title: "Fused softmax"
date: 2025-06-12
draft: true
description: ""
slug: "fused-softmax"
tags: ["gpu-programming", "triton"]
---

# 

> Note: this tutorial is taken from the [official Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html#sphx-glr-getting-started-tutorials-02-fused-softmax-py)

Let's start by a fact: GPU programming is hard. I'm not entirely sure why it's that hard to write kernels: maybe because they're technically challenging? Or perhaps it has to do with the way GPUs work, which is very much different of how us humans think. In any case, writing kernels is hard.
Furthermore, using CUDA as a language can, for some people, increase its complexity, mostly because people involved in Machine Learning are typically more comfortable in writing Python code. Writing a kernel in CUDA and C++ also poses other complexities, like how to actually build the software, how to integrate it correctly in a Python program, how to debug it quickly etc.

To reduce this complexity,  was made by OpenAI, circa 4 years ago. The idea is very simple: write GPU kernels in pure Python. That seems like a dream, right? Well, wake up, because it's reality and today we are going to learn Triton's basics by implementing a Fused Softamx.


Let's begin with a fundamental truth: GPU programming resembles navigating an alien computational landscape. The difficulty isn't merely technical complexity. It stems from the profound mismatch between human sequential thinking and GPU parallel architecture. Our minds construct logical narratives step-by-step, while GPUs orchestrate thousands of simultaneous operations, each following identical instructions on different data fragments.
CUDA amplifies this challenge for machine learning practitioners. Beyond mastering C++ syntax, developers must navigate memory hierarchies, thread synchronization, and optimization patterns; then integrate these kernels into Python workflows, manage build systems, and debug performance bottlenecks across entirely different computational paradigms.
To bridge this divide, OpenAI introduced [Triton](https://openai.com/index/triton/) four years ago with an elegant proposition: write GPU kernels in pure Python. This isn't syntactic sugar over CUDA; it's a philosophical shift toward democratizing GPU programming, allowing practitioners to express parallel algorithms using familiar constructs while maintaining hand-optimized performance.
Today, we'll explore this computational intersection by implementing a Fused Softmax, a pattern that exemplifies both the challenges and opportunities in modern GPU programming.

So, wash your hands and put on some gloves.


### Before starting
As always, there are some things before we can start getting our hands dirty.

The first thing you must know is how a GPU works. There are a ton of great resources online, so let me suggest you one of mine: [Cerebral Cortex and Hippocampus: Understanding the Computational and Memory Design of GPUs](https://themlsurgeon.substack.com/p/cerebral-cortex-and-hippocampus-understanding). Concepts like SMs, memory hierarchies, occupancy, threading etc. are fundamental knowledge for GPU programming, so you really cannot skip this part.

Additionally, I recommend starting with basic CUDA knowledge. Again, here's my take: [Hello CUDA: A Surgical Dissection](https://themlsurgeon.substack.com/p/hello-cuda-a-surgical-dissection). Understanding CUDA before learning Triton proved invaluable for me. It's like mastering C++ before transitioning to Python. Everything becomes clearer, and you understand the underlying computational machinery

If you're already familiar with CUDA, grasp this crucial difference: CUDA's programming model centers on thread blocks operating on scalars. When accessing matrix elements, you index directly: `A[m, k]`. Triton inverts this paradigm: you work with blocked programs where operations target data segments, like this: `A[m:m+MB]`.
This shift fundamentally changes kernel conceptualization. Instead of orchestrating individual threads, you choreograph block-level operations. This blocked programming model naturally aligns with neural network computation patterns, making kernel development significantly more intuitive for machine learning applications.



## What is a Fused Softmax
If you're reading this blogpost, I assume you know what Softmax is, but just in case, here's a quick overview.

The Softmax function is defined as:

{{< katex >}}
\\(Softmax(z) = \frac{e^{z_i}}{\sum_{j=1}^{K}{e^{z_j}}}\\)

This mathematical expression simply means that each element of our vector gets set to a value representing the ratio between its exponential and the sum of all exponentials in the vector. This proves incredibly useful because it transforms logits (raw scores) into proper probabilities. The Softmax function produces a probability distribution where each element represents the probability of the element at position i. As you can imagine, this becomes essential for classification problems.

However, we won't dive deep into the mathematical theory today. Instead, we'll focus on how Softmax can be naively implemented in PyTorch and discover why that approach makes absolutely no sense from a performance perspective.

```python
def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret
```

As you can see from this code snippet ([taken from here](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html#sphx-glr-getting-started-tutorials-02-fused-softmax-py)), we perform a total of `5MN + 2M` element reads and `3MN + 2M` element writes. Since moving data around the GPU is computationally expensive, all these memory transfers between global memory and processing cores represent a significant performance bottleneck.

This introduces the concept of `kernel fusion`, a fundamental optimization technique used throughout modern GPU computing. The principle is elegantly simple: read data once, perform all necessary computations on-chip, then write the final result back to memory. This approach dramatically reduces the memory bandwidth requirements that often limit GPU performance.

Here's a visual comparison between the two computational approaches:
[PUT GRAPHICS HERE]

What we're going to do next is precisely this: write a Triton kernel that efficiently computes the softmax function. However, there's an important constraint: our implementation will only handle cases where input row sizes fit within the kernel's block size. Keep this limitation in mind as we proceed! What exactly is a block size? You'll discover that shortly.


## Hands on the kernel
