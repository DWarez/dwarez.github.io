---
title: "Move Fast or Die Slow"
date: 2025-02-01
draft: false
description: ""
slug: "speed-or-die"
tags: []
---

> This will not be an hyper-technical article like the ones I use to write. Therefore, if you only care about tech and code, feel free to skip.


1. Start from abstraction (having a better model [how to define better?], having a nicer user experience [what is a good user experience in the context of AI?])
2. These things are connected to the optimization of the ML stack. Speed for training and inference.
3. Speed in training = more experiments -> better models -> can experiment faster than competitors and find the best novelty (see DeepSeek with their MLA and other techniques)
4. Speed in inference = better user experience -> your product is faster overall because typically the responsiveness bottleneck is created by models
5. Speed is time -> GPU hours as a metric for calculating costs -> More efficiency = less cost = you get more out of your infrastructure, you can serve at a lower price point etc.


The AI landscape is evolving at a breakneck pace, and conventional wisdom suggests that competitive advantage is simply a function of computing power - the more GPUs you can afford, the better positioned you are to win. However, recent events in the market tell a different, more nuanced story.
Consider what happened when DeepSeek released their R1-Zero and R1 models. The announcement sent shockwaves through the US stock market, affecting even giants like NVIDIA. Why would the release of two models from a relatively smaller player have such a dramatic impact? The answer lies in what these models represented: a fundamental challenge to the "more hardware equals better results" paradigm.
Operating under China's GPU restrictions, DeepSeek demonstrated something remarkable - they achieved results comparable to industry leaders while reportedly using significantly less computational resources. While the full technical details and production costs remain a subject of debate (and I encourage readers to check [Semianalysis's detailed breakdown](https://semianalysis.com/2025/01/31/deepseek-debates/) for a fact-based perspective), the key takeaway isn't about DeepSeek specifically. It's about what their approach represents: a shift from brute-force scaling to intelligent optimization.
This paradigm shift reveals a crucial truth about today's AI market: raw computational power alone isn't enough. The real competitive edge comes from how effectively companies can optimize their AI infrastructure and development pipeline. The winners in this space won't necessarily be those with the biggest GPU farms, but those who can:

1. Execute more meaningful experiments in less time
2. Deliver faster inference speeds for better user experiences
3. Maximize the efficiency of their infrastructure investments

In this article, we'll explore how these factors interplay and why they matter more than ever in today's hyper-competitive AI landscape.


## The Dual Benefits of Optimization
In the AI industry, optimization isn't just a technical nicety—it's a strategic imperative that creates value for both companies and users. Let's break down why this matters from both perspectives.
For users, an AI product's value proposition is remarkably straightforward: it needs to be both intelligent and fast. Intelligence is non-negotiable, it's the core promise of AI, especially in generative applications. However, raw intelligence isn't enough. A brilliantly capable model that takes too long to respond creates a frustrating user experience. In fact, there's likely a critical response-time threshold (though its exact value remains debatable) beyond which users will abandon a smarter but slower solution in favor of a faster, less capable alternative.
For companies, developing such products presents a complex challenge. While the definition of a "good" AI product might be simple, intelligent and fast, achieving this balance within real-world constraints is extraordinarily difficult. Companies must navigate limited budgets, finite computational resources, market timing pressures, and funding constraints.
Here's where it gets interesting: these constraints, rather than being purely limiting factors, often drive innovation. Consider this paradox: companies with unlimited resources often innovate less in terms of efficiency. When you can solve any problem by throwing more hardware at it, optimization becomes an afterthought. Similarly, companies with massive funding might neglect infrastructure optimization, believing they can simply buy their way out of performance bottlenecks.
But this approach has a critical flaw: it eventually hits a wall. The recent market dynamics between OpenAI and DeepSeek illustrate this perfectly. When scaling through raw computing power reaches its limits, companies that haven't invested in optimization find themselves vulnerable to more efficient and innovative competitors. The ability to do more with less becomes a crucial competitive advantage.

## Breaking Down DeepSeek's Success: Speed in Training and Inference
DeepSeek's approach demonstrates two critical areas where optimization creates competitive advantage: accelerating experimentation through efficient training, and delivering better user experience through optimized inference. Let's examine both.

### The Experimentation Advantage
In AI development, there's no substitute for experimentation. Despite machine learning's mathematical foundations, the behavior of models at scale often surprises even the most experienced experts. Success comes through systematic trial and error, making the speed of experimentation a crucial competitive factor. Simply put: the company that can run more meaningful experiments faster has a higher probability of achieving the next breakthrough.
DeepSeek exemplified this by optimizing their training pipeline to maximize experiments per unit of infrastructure. Their strategic choice to explore Reinforcement Learning over traditional Supervised Fine Tuning (SFT) wasn't just about building a smarter model—it was about finding a more efficient path to innovation. By focusing on training workflow efficiency, they could conduct more experiments despite infrastructure constraints, ultimately leading to novel methodologies.

### Making Intelligence Fast
However, developing a smart model is only half the battle. DeepSeek's R1 model uses Chain of Thought reasoning (DeepThink) to achieve high accuracy, but this approach requires generating significantly more tokens than models like Claude Sonnet. Without optimization, this would result in unacceptably long response times—potentially minutes of waiting for users.
DeepSeek tackled this challenge through several innovative approaches. At the core of their optimization strategy is the Multi-head Latent Attention, which achieved a remarkable 93.3% reduction in KV Cache requirements per query. They further enhanced performance through multi-token prediction and quantization techniques. The choice of a Mixture of Experts (MoE) architecture means fewer parameters need to be activated during inference, making the model more efficient to run.
These optimizations transform directly into business value. The reduced computational requirements mean lower infrastructure costs and more competitive API pricing. Most importantly, they enable DeepSeek to deliver what users truly want: intelligent responses that arrive quickly. The optimization work ensures users aren't forced to choose between smart answers and fast ones—they get both.




Deepseek = MLA, and stuff
OpenAI = let's add more data

DeepSeek = RL instead of SFT
OpenAI = why shouldn't we SFT? Let's add more data

DeepSeek = more thinking -> more tokens -> better results -> possible thanks to MLA, multi token prediction, latent kv cache 
OpenAI = RAG in the chat, instead of having a huge context window like Claude -> openai is starting to lose ground

DeepSeek = write in PTX, no dependency on nvidia, can run inference on Huawai chips
OpenAI = triton, so even more abstract that CUDA, which means adding a dependency

In general: optimizing makes you spend less money, less time, be faster, more agile, more versatile. It's harder to do w.r.t. scaling with hardware or data, but it gives you better results.

Restricted markets (like defense, healthcare) often rely on embedding systems, so running large models on low power is another interesting topic.
