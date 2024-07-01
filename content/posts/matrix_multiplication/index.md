---
title: "Matrix Multiplication"
date: 2024-06-25
draft: true
description: ""
slug: "matrix-multiplication"
tags: ["cuda", "gpu", "architecture"]
---

During the first year of my Master's Degree in Computer Science, I had to complete a project for a Machine Learning course. The project involved implementing a small feed-forward neural network framework from scratch, using only numerical libraries and coding elements such as loss functions, backpropagation, and the feed-forward step.

That project was crucial for me because it revealed an inconvenient truth: matrix multiplication is the most important aspect of Machine Learning. Hence, I named my project "ML is ML": Machine Learning is Matrix Multiplication. Although the course professor didn't fully appreciate the title, it still earned me the highest grade. Because deep down, he knew it was true. It was, indeed, an inconvenient truth.

> **_NOTE:_**  Yes, I'm well aware that tensor contraction is not the same as matrix multiplication. Still, the "ML is ML" joke only works when talking about matrices, so stick with it.

However, this is not the place to explain why matrix multiplication is so fundamental to modern AI. Besides, if you're reading this blog, you probably already know.

Instead, as Machine Learning Surgeons, we should ask ourselves how such an important operation is implemented to fully utilize the power of GPUs. It's easy to implement something, but it's much harder to make it run fast! Just like humans need to train their reaction times to do tasks like driving an F1 car, we Machine Learning Surgeons must operate on the muscle fibres of kernels in order to make them fast and powerful!

So, put on your gloves, wield your scalpels and let's start the operation!


## The starting point
Since you're still a pratictioner, I'll walk you through the simplest matrix multiplication kernel.
But first, there is a basic concept that you have to grasp before proceeding with the code.


### Matrix linearization
As you may have learned from the [Hello CUDA](https://dwarez.github.io/posts/hello-cuda/) article, when writing CUDA kernels, we typically use block and thread indices to select elements for computation. But what happens when dealing with higher-dimensional data structures, such as matrices? Moreover, how is memory organized on GPUs when dealing with such data structures?

In CUDA, memory on the device is managed linearly, meaning it is stored as a single, contiguous block of memory. Therefore, matrices are stored in this contiguous block of memory in a row-major order. Linearization involves mapping the multi-dimensional indices of a matrix to a single-dimensional index. It's much easier to do than to explain.

Consider a matrix {{< katex >}}\\(\mathrm{A} \in \mathrm{R}^{\mathrm{N} x \mathrm{N}}\\)
stored in a row-major order, which is the representation used by CUDA. The linear index of the element {{< katex >}}\\(\mathrm{A}\_{i, j}\\) (i-th row and j-th column of {{< katex >}}\\(\mathrm{A}\\)) is simply given by {{< katex >}}\\(i * \mathrm{N} + j\\). This is because in the row-major representation, the array that represents the 2D matrix is a continuous sequence of the rows of the matrix. Therefore, to acces the item {{< katex >}}\\(\mathrm{A}\_{i, j}\\) we must skip {{< katex >}}\\(i\\) rows, doing {{< katex >}}\\(i * \mathrm{N}\\) and sum to it the column index {{< katex >}}\\(j\\).


### The simplest kernel
Great! Now you are ready to fully understand the simplest matrix multiplication kernel. Keep in mind that I'll avoid writing all the usual boilerplate code for instantiating variables on the host, moving data to the device, and printing results. If you have any doubts, you can check the full code here: TODO link {{< icon "github" >}}

```c {linenos=true}
__global__ void matrixMulKernel(float* C, float* A, float* B, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float value = 0;
        for (int k = 0; k < N; ++k) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}
```
You should be able to understand most of this code, but let's quickly walk through it.

In lines 2-3, we define the index of the row and column of the matrix element that the thread will use for the dot product computation. Remember that the mathematical notation and the CUDA representation are inverted. While in mathematical notation we use the x-axis for referring to the rows and the y-axis for referring to the columns, in CUDA we use the y-components to compute the row index and the x-components to compute the column index.

Line 5 simply checks the index boundaries. We must check this condition because we will likely spawn more threads than there are elements in the matrix, so some threads may compute out-of-bounds indices.

Lines 6-10 perform the actual dot product computation. We start by initializing a local variable to accumulate the result. Given that this is a local scalar variable, it will be stored in a register, the fastest type of memory available. This approach works well because the dot product, which involves summing the products of corresponding elements, can be computed incrementally. Remember this, as it will be essential for the upcoming optimization steps!

Line 7 defines the loop that allows the thread to iterate over all elements of the rows of A and the columns of B. In each iteration, we accumulate the value by adding the product of the linearized row element from A and the linearized column element from B. If you find it difficult to visualize the linearizations, I suggest writing out a few loop iterations on a piece of paper.

Lastly, line 10 stores the computed dot product into the linearized element of C.

For completeness, I will also include the kernel invocation. Keep in mind that you can use the `ceil()` function for the `gridSize`

```c
dim3 blockSize(16, 16);
dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
matrixMulKernel<<<gridSize, blockSize>>>(d_C, d_A, d_B, N);
```

### Analysis and limitations
Now that we know each 