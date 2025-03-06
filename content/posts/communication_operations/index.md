---
title: "Comms"
date: 2025-03-06
draft: false
description: ""
slug: "comms"
tags: ["hpc"]
---
During my Master's degree in Artificial Intelligence and Machine Learning at the University of Pisa, I took a variety of courses. These ranged from highly specialized subjects (machine learning, human language technologies, systems for pattern recognition) to more foundational ones (algorithm engineering, information retrieval, computational mathematics).

Like many enthusiastic students, I was drawn to the flashy, specialized courses while undervaluing the fundamental ones. I was young and naive, not yet realizing that understanding vertical, technical topics requires a solid foundation in algorithmic basics. This oversight is a common trap in technical education.
One course I particularly disregarded was Parallel and Distributed Computing Systems. At the time, it seemed tedious compared to the allure of models, optimization, and the "wow factor" of artificial intelligence. GPU architectures and parallel design patterns? Boring! I wanted the glamour of AI, not the plumbing.

Yet as I delved deeper into AI, I had an epiphany: what appears magical is simply what we don't yet understand. Yes, modern AI systems have remarkable capabilities that can seem almost intelligent. But creating such systems isn't mystical, it's methodical, provided you understand the fundamentals. A newcomer might envision these massive models as mysterious entities that somehow absorb knowledge and develop intelligence. An AI practitioner, however, recognizes that behind the curtain, it's primarily about computational scale and efficiency.

Today, what fascinates me most is precisely this pragmatic reality of AI systems. Opening up the "patient" and examining the "organs" of these systems has become intellectually stimulating in ways I couldn't have anticipated as a student.

So why share this journey? Because many Machine Learning Engineers take for granted how we scale these enormous systems and computations. We use abstracted software tools, add hardware, and, seemingly like magic, our giant model runs. But what's actually happening under the hood?

To answer this complex question, we need to start with the fundamentals. This brings us to communication operations, the circulatory system of machine learning parallelization that makes large-scale AI possible.


## Why we care, what's the deal

Before diving into technicalities, let's establish why understanding communication operations and patterns is crucial in AI systems.

While I won't explore parallelization methodologies exhaustively in this article, consider this simple scenario: you have two GPUs, each capable of hosting your model. How can you leverage both GPUs working in parallel? A common approach is Data Parallelism, where identical copies of the model exist on each GPU, and your dataset is divided between them. This allows parallel computation with a larger effective batch size. If one GPU can train with a batch size of 16, two GPUs enable a global batch size of 32, roughly linear scaling.

However, a fundamental challenge emerges: how do we maintain consistency between model replicas? Without communication, each GPU would train a different model, defeating our purpose. We want system components to cooperate, enhancing training efficiency. This cooperation necessitates communication, perhaps sharing gradients computed by each GPU so all replicas update their parameters with the same information.

This example illustrates just one scenario requiring inter-component communication. The complexity increases dramatically when we scale to cluster environments with numerous nodes. How do we orchestrate communication across such distributed systems?

Communication presents two major challenges. First, implementing it efficiently is technically difficult. Second, it introduces overhead. In some cases, this overhead can be so substantial that it negates the benefits of parallelization. The key insight is to implement efficient communication and, critically, overlap it with computation. This creates a pipeline where one part of the system computes while another simultaneously transfers data.

This overlap becomes even more vital in Machine Learning, where we transmit massive volumes of data: layers, parameters, checkpoints, optimizer states, and more. Additionally, as we increase the number of GPUs, the communication complexity grows exponentially.

While we'll address the specific hardware involved in this process in a future article, I hope you now appreciate the significance of communications in AI systems, especially when training large-scale models. With this foundation established, let's examine the fundamental operations that make this communication possible.