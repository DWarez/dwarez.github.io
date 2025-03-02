---
title: "Scaling"
date: 2025-03-01
draft: false
description: ""
slug: "scaling-laws"
tags: ["llm"]
---


### Scaling Laws
"Scaling Laws for Neural Language Models" is a paper from OpenAI published in 2020. In this effort, Kaplan et al tried to answer one of the most interesting questions in Machine Learning: given a model and a dataset, which result should I expect? If there existed a closed formula to obtain an estimate of the end result of a training without performing it, we would save so much money. At this point in time, it's fundamentally impossible to know a priory the result of an experiment; however this paper resulted in a very interesting result, which formally makes an estimation of the expected Loss of the model, given its size (in terms of parameters), the size of the dataset used to train it (in terms of tokens) and the computational power used to train it (in terms of FLOPs).

The discovery is that the relashionship between these componets can be described as a power law. 

If {{< katex >}}\\(N\\) is the number of parameters of the model, then the loss {{< katex >}}\\(\mathcal{L}\\) can be estimated like:

{{< katex >}}
\\(\mathcal{L}(N) \propto N^{-\alpha}\\)

where {{< katex >}}\\(\alpha\\) is approximately 0.076