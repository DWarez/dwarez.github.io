---
title: "Hello CUDA: the righteous way"
date: 2024-05-06
draft: true
description: "The CUDA programming language enables developers to harness the power of NVIDIA GPUs for general-purpose processing tasks. Given its steep learning curve, many."
slug: "hello-cuda"
tags: ["gpu-programming", "cuda"]
---

In case you didn't already know, [CUDA](https://it.wikipedia.org/wiki/CUDA) is a parallel computing platform and API, developed by NVIDIA, that enables programmers to exploit certain types of GPUs. Not too long ago (around until 2007), GPUs were solely utilized for graphics rendering, and programmers had to depend on very specific graphics APIs to utilize them for solving general problems.

With the introduction of CUDA, everything changed. Its integration with languages like C, C++, or Python, along with the elimination of the need for knowledge in graphics programming, made GPU programming much more accessible.

However, many individuals today still feel daunted by the complexity of GPU programming, both in terms of the language (which is too low-level compared to the most popular programming languages today) and the architecture and development knowledge required to write this kind of software.

This article aims to introduce you to this realm through an in-depth technical analysis of a HelloWorld, which is famously the starting point for every programmer who ventures into a new programming language.

Since we're discussing GPU programming, the classic `print("hello world")` won't be applicable. Instead, we'll analyze a complete program that adds two vectors, using the GPU.


## Prerequisites
I will briefly present some prerequisites necessary to fully understand the remaining content of this post. Even if you don't meet all of them, I still suggest you keep reading. I'm confident you can still enjoy the content and find it useful.

- Basic C/C++ knowledge. Given that CUDA is originally a superset of C/C++, it comes natural that you have to know the basics of these languages.
- Basic knowledge of a GPU architecture.  I won't examine deeply this topic here, but understanding how a GPU works and its computational paradigm is essential for grasping CUDA programming.
- Installation of CUDA on your machine to try out the code yourself.

That's about it.


## Code Analysis
In the link below, you will find the full code snippet for our HelloWorld program. I believe that reading the entire code first piques your interest, as you will likely have questions you want answered.

If this code seems intimidating to you, please don't be afraid. I assure you that by the end of your reading, you will fully understand it and realize its simplicity.

TODO: inserisci link a Github gist {{< icon "github" >}}

We are now ready to analyze this code, top to bottom.


### CUDA Error Handling
The code example begins with the definition of a macro, which will be utilized throughout the program.

```c
#define CUDA_CHECK_ERROR(err) \
    if (err != cudaSuccess) { \
        printf("CUDA err: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    }
```

Within this macro, we utilize several CUDA-defined functions and symbols, such as `cudaSuccess` and `cudaGetErrorString(cudaError_t)`. While these are fairly self-explanatory, let's delve deeper into their significance.

We consider the `err` variable a `cudaError_t` type variable, intended to store errors encountered during execution. Within the macro, we check if the error status is `cudaSuccess`, indicating successful code execution. If this condition is not met, we invoke the `cudaGetErrorString(cudaError_t)` function to retrieve the error string, providing clarity on the encountered issue, and then we exit the code execution.

While this is obviously the most basic way of doing error handling, it's crucial to remember to remember its necessity whenever utilizing CUDA APIs, as errors may arise during their invocation.


### The Kernel
Now, we'll delve into what is arguably the most intriguing segment of the code, which will unveil some pivotal technical insights into CUDA programming and hardware.

To begin, it's essential to understand that the term "kernel" typically denotes functions that can be computed by the device. In this context, we're referring to the code responsible for orchestrating the GPU's addition of the two input vectors.

Let's now examine the function's prototype.

```c
__global__ void vecAddKernel(float* A, float* B, float* C, int n)
```

If you are familiar with the C programming language, you may have noticed that the keyword `__global__` doesn't come from the standard. It is in fact a keyword implemented by the CUDA programming language, which specifies the visibility of the function.

#### Function Execution Space
In the GPU programming paradigm, we distinguish between two entities: the host and the device. The host refers to the CPU along with its memory, while the device pertains to the GPU. Consequently, the CUDA programming language incorporates three function execution specifiers:

- `__global__` denotes a kernel. The function is executed on the device, and is callable both by the host and the device (only for devices of compute capability 5.0 or higher). It's crucial to note that functions of this type must return void.
- `__device__` signifies code executed exclusively on the device. It can only be called by another device execution and not from the host.
- `__host__` denotes code that can be called and executed solely by the host. This is the default execution space if no execution space is specified.


Now that we understand what a function execution space entails, let's briefly delve into the remainder of the prototype. As mentioned earlier, given that this function serves as a kernel, it must inherently return `void`. Consequently, we'll need both the input and output vectors, as well as explicitly specify the length of the vectors.


#### Blocks and threads
The subsequent lines of code may appear cryptic at first glance, as it encompasses several complexities. To unravel its intricacies, let's begin with the fundamentals.

```c
int i = threadIdx.x + blockIdx.x * blockDim.x;
if (i < n) {
    C[i] = A[i] + B[i];
}
```

This article doesn't delve deeply into GPU architecture, so I won't discuss its overall structure here. However, it's crucial to understand that GPUs operate on the [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) (_Single Instruction Multiple Data_) paradigm. This allows GPUs to process multiple data in parallel, executing a single instruction.

In essence, the heart of GPU processing lies in threads. The kernel we're defining will be executed by the device, which will use a certain number of threads to execute the kernel instructions.

Let's illustrate this with a practical example using the code snippet above. Suppose the length of both vectors is 100. We'll write code to execute the kernel, allocating 100 threads. Each thread will be responsible for summing a specific position in the vectors.

In the code snippet, the position is indicated by the variable `i`. It's crucial for each thread to have a distinct `i` value; otherwise, all threads would operate on the same position in the vectors. Fortunately, the CUDA API provides built-in thread-specific variables that help us compute the correct positional index. Specifically, we're utilizing the following built-ins:

- `threadIdx`: Represents the index of the thread currently executing the code. 
- `blockIdx`: Denotes the index of the block containing the executing thread.
- `blockDim`: Specifies the size of each spawned block.


