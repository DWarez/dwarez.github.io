---
title: "Hello CUDA: A Surgical Dissection"
date: 2024-05-06
draft: false
description: "The CUDA programming language enables developers to harness the power of NVIDIA GPUs for general-purpose processing tasks. Given its steep learning curve, many."
slug: "hello-cuda"
tags: ["gpu-programming", "cuda", "basics"]
---

In case you didn't already know, [CUDA](https://it.wikipedia.org/wiki/CUDA) is a parallel computing platform and API, developed by NVIDIA, that enables programmers to exploit certain types of GPUs. Not too long ago (around until 2007), GPUs were solely utilized for graphics rendering, and programmers had to depend on very specific graphics APIs to utilize them for solving general problems.

With the introduction of CUDA, everything changed. Its integration with languages like C, C++, or Python, along with the elimination of the need for knowledge in graphics programming, made GPU programming much more accessible.

However, many individuals today still feel daunted by the complexity of GPU programming, both in terms of the language (which is too low-level compared to the most popular programming languages today) and the architecture and development knowledge required to write this kind of software.

This article aims to introduce you to this realm through an in-depth technical analysis of a HelloWorld, which is famously the starting point for every programmer who ventures into a new programming language.
Just as a surgeon carefully dissects to reveal the inner workings of the human body, we'll break down each component of CUDA's basic architecture and functionality.

Since we're discussing GPU programming, the classic `print("hello world")` won't be applicable. Instead, we'll analyze a complete program that adds two vectors, using the GPU.


## Prerequisites
I will briefly present some prerequisites necessary to fully understand the remaining content of this post. Even if you don't meet all of them, I still suggest you keep reading. I'm confident you can still enjoy the content and find it useful.

- Basic C/C++ knowledge. Given that CUDA is originally a superset of C/C++, it comes natural that you have to know the basics of these languages.
- Basic knowledge of a GPU architecture.  I won't examine deeply this topic here, but understanding how a GPU works and its computational paradigm is essential for grasping CUDA programming.
- Installation of CUDA on your machine to try out the code yourself.

That's about it.


## Code Dissection
In the link below, you will find the full code snippet for our HelloWorld program. I believe that reading the entire code first piques your interest, as you will likely have questions you want answered.

If this code seems intimidating to you, please don't be afraid. I assure you that by the end of your reading, you will fully understand it and realize its simplicity.

[Here's the code](https://gist.github.com/DWarez/90515ace919f5dca6e65a4d35f09a8b5) {{< icon "github" >}}

We are now ready to dissect this code, top to bottom.


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

While this is obviously the most basic way of doing error handling, it's crucial to remember its necessity whenever utilizing CUDA APIs, as errors may arise during their invocation.


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


Now that we understand what a function execution space entails, let's briefly delve into the remainder of the prototype. As mentioned earlier, given that this function serves as a kernel, it must inherently return `void`. Consequently, we'll need both the input (`A`, `B`) and output (`C`) vectors, as well as explicitly specify the length of the vectors (`n`).


#### Blocks and threads {#blocks_and_threads}
The subsequent lines of code may appear cryptic at first glance, as they encompasses several complexities. To unravel their intricacies, let's begin with the fundamentals.

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

A block is a group of threads that collaborate and communicate through shared memory. These blocks are scheduled and executed together by the hardware. Moreover, all blocks together form what is termed as a grid. This implies that the structure is inherently multi-dimensional: a grid has three dimensions—x, y, and z. Hence, when referencing the built-in variables `threadIdx`, `blockIdx`, and `blockDim`, we utilize their `x` attribute, which corresponds to the x-dimension of the grid. The rationale behind this will become clearer when we examine the kernel invocation.

The final piece to explain is the `if` statement preceding the actual sum computation. This if statement verifies that the computed index falls within the bounds of the input vectors. While this check may initially seem unnecessary, its significance will become apparent later on.


### Host code
Following the discussion on the kernel, let's shift our attention to the host code of the program. Here, we're tasked with memory allocation, kernel invocation, and error checking.

#### Device Memory
A crucial consideration is that device memory is distinct from host memory. Consequently, the device cannot directly operate on data residing in host memory. This necessitates the allocation of device memory and the transfer of data from the host to the device. While this may appear trivial, it's one of the most significant bottlenecks in modern software, particularly when handling large amounts of data, leading to substantial communication overhead between the host and the device.
Given this distinction, as a best practice, we use the notation `V_h` to indicate a variable used by the **h**ost and `V_d` to denote a variable used by the **d**evice.

Given the simplicity of our case study, we'll overlook these technical intricacies and focus on accomplishing the bare minimum to ensure functionality.

```c
void vecAdd(float* A_h, float* B_h, float* C_h, int n)
```

The `vecAdd` function accepts three host-vectors as arguments: `A_h` and `B_h`, which are to be summed, and `C_h`, which will store the result. It's important to note that an output vector `C` is necessary since the kernel cannot directly return anything. Finally, `n` represents the length of the input vectors.

Our initial step is to determine the size of the device-vectors. This computation follows a standard C approach:

```c
int size =  n * sizeof(float)
```

Next, we proceed to allocate memory on the device to store the device-vectors:

```c
float *A_d, *B_d, *C_d;
cudaMalloc((void**)&A_d, size);
cudaMalloc((void**)&B_d, size);
cudaMalloc((void**)&C_d, size);
```

Fortunately, the CUDA programming language provides us with the `cudaMalloc` function, which functions similarly to the `malloc` function but operates on the device.

Before invoking the kernel, the final step is to transfer data from the host to the device. At this stage, we have:
- On the host, three vectors containing the data for computation
- On the device, three memory allocations initialized but lacking of the necessary data.

To address this, we leverage another CUDA programming function, `cudaMemcpy`:

```c
cudaError_t error = cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
CUDA_CHECK_ERROR(error);
error = cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
CUDA_CHECK_ERROR(error);
```

We invoke the function with the following parameters: the destination pointer, the source pointer, the size to copy, and the direction. The parameter `cudaMemcpyHostToDevice` specifies that we intend to copy data from the host to the device.
Additionally, observe how we utilize the previously defined macro to check for errors in case the memory copying operation fails.

#### Kernel Invocation
Assuming everything executes correctly, we've successfully copied the input vectors to the device memory. Hence, we can proceed to call the kernel.

```c
int number_of_threads = 256;
dim3 dimGrid(ceil(n/number_of_threads), 1, 1);
dim3 dimBlock(number_of_threads, 1, 1);
vecAddKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, n);
```

First, we define the number of threads we want to allocate per block. Next, we define two `dim3` type variables, which represent 3-dimensional vectors typically used to specify grid and block sizes. Since we are dealing with a 1-dimensional problem, we only need to use the x-dimension.

We compute the grid size as the number of elements in the input vectors, `n`, divided by the number of threads per block. To ensure the grid dimension can accommodate all threads, even if `n` is not perfectly divisible by the number of threads per block, we use the `ceil` operator, which will round-up the division. The block dimension is simply set to the number of threads we want to allocate per block.

Lastly, we invoke the kernel, as specified in the previous sections. 

Notice two things:

1. We can now fully understand why we need to use an `if` statement in the kernel, as explained in the [Blocks and threads](#blocks_and_threads). Let's illustrate with a practical example. Suppose `n = 100` and `number_of_threads = 16`. When computing the x-dimension of the grid size, the exact result is 6.25. However, if we set the grid x-size to 6, we won't allocate enough threads because we'll have 6 blocks of 16 elements each, totaling 96 threads, leaving out 4 positions of the vectors. On the other hand, if we set the grid size to 7, we'll allocate more threads than the number of positions to compute because 7x16=112. We can't allow the extra threads to access memory indicated by their `int i = threadIdx.x + blockIdx.x * blockDim.x;` because it would be out of bounds. Therefore, the necessity of using an `if` statement to check such boundaries.
   
2. Often, tutorials do not explicitly allocate the `dimGrid` and `dimBlock` variables, but instead invoke the kernel directly like this: `vecAddKernel<<<ceil(n/number_of_threads), number_of_threads>>>(A_d, B_d, C_d, n)`. This syntax is valid because the `dim3` data type automatically defaults unspecified dimensions to 1. However, I chose to specify all dimensions to clarify the explanation provided in the section [Blocks and threads](#blocks_and_threads).


#### Storing and cleaning up
After the kernel invocation, we need to retrieve the computation result. Since the kernel cannot return anything, it's the programmer's responsibility to fetch the output of the computation. We can easily accomplish this as follows:

```c
error = cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
CUDA_CHECK_ERROR(error);
cudaFree(A_d);
cudaFree(B_d);
cudaFree(C_d);
```

Once again, we utilize the `cudaMemcpy` function, but this time we specify `cudaMemcpyDeviceToHost`, indicating that we are transferring data from the device to the host. Notice also how the order of the host and device variables has switched in the function call compared to previous occurrences.

Finally, we must free the memory that was used. This isn't just a best practice—it's an absolute necessity!


### Main
Just for completeness, let's quickly go through the main function:

```c
    int n = 512;
    float A_h[n], B_h[n], C_h[n];
    for (int i = 0; i < n; i++) {
        A_h[i] = i;
        B_h[i] = i;
    }
    vecAdd(A_h, B_h, C_h, n);
    for (int i = 0; i < n; i++) {
        printf("%g\n", C_h[i]);
    }
    return 0;
```
There isn't much to say here. We define the length of the vectors `n`, then allocate and initialize the input vectors `A_h` and `B_h` with a range from 0 to n-1. Next, we call the host code, which also invokes the kernel, and finally, we print the result stored in `C_h`.


## Compile and run
Compiling and running this example is straightforward. First, we compile the source as we would for a C file, but we use the CUDA compiler, called `nvcc`. Assuming the source is named `hello_world.cu` and we are in its parent directory, we can build the executable like this:

```sh
nvcc hello_world.cu -o hello_world
```

Then, we simply run the executable:

```sh
./hello_world
```

And we will receive the output (truncated for readability):

```sh
0
2
4
...
1018
1020
1022
```

## Wrapping Up
In summary, we've explored the fundamentals of CUDA programming, which enables us to harness the power of GPUs for parallel computing tasks. We began by understanding the distinction between the host (CPU) and the device (GPU) and delved into CUDA's function execution specifiers (`__global__`, `__device__`, `__host__`) that define where functions can be executed.

We discussed the importance of memory management, highlighting how data must be transferred between host and device memories using functions like `cudaMalloc` and `cudaMemcpy`. We also saw the significance of error handling throughout the CUDA programming process, utilizing macros to check for errors and ensure smooth execution.

The heart of CUDA programming lies in kernels, functions executed on the GPU, typically invoked in a grid of threads. We learned about grid and block dimensions, and the necessity of boundary checks within kernels to avoid out-of-bounds memory access.

We then examined a practical example of vector addition, illustrating how to define and invoke kernels, manage memory, and handle errors.

Lastly, we discussed compiling and running CUDA programs using the `nvcc` compiler and executing the resulting executables.

Overall, CUDA provides a powerful framework for parallel programming on GPUs, offering immense computational capabilities for a wide range of applications. With a solid understanding of its principles and techniques, developers can unlock the full potential of GPU-accelerated computing.

Time to close the operating theater! You've witnessed the basic intricacies of CUDA programming under the scalpel today. Keep honing those skills, and you'll soon be performing GPU surgeries with finesse. Until our next exploration, stay sharp and happy coding! 🤖