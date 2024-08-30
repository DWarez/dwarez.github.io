---
title: "A quick incision: ten minutes to RAG"
date: 2024-08-29
draft: false
description: ""
slug: "ten-minutes-to-rag"
tags: ["rag", "llm", "vector-db"]
---

Hello, fellow surgeons! How is life treating you? I hope you've spent your vacation relaxing, far, far away from the tools of our trade. After all, a good surgeon needs to rest after a long year of work and learning, right? With that in mind, I've chosen a simple yet useful topic to discuss today, so you can stay relaxed and not worry about the tremendous complexity of our field—at least for now.


I'm sure you've come across the term **RAG** at least once. It stands for Retrieval-Augmented Generation, a technique that's become quite popular these days. Its popularity stems from the fact that RAG systems are relatively simple to implement yet highly effective in terms of performance, all while keeping infrastructure **costs reasonably low**.

Don't worry, you won't need your gloves today. This incision will be quick and easy, and we won't go too deep—no risk of getting blood on you!


## Prepping the Instrument: Understanding RAG
Imagine this scenario: you're a med student who has, over the years, compiled a vast set of notes from your courses, covering all the topics you've studied. You’re confident that this material will help you become a great surgeon! But there's a problem: the sheer volume of this **knowledge base** makes it difficult to query. Even with well-organized notes, pinpointing the exact information you need can be a long and tedious task.

Pressed for time and with many questions that need answering, you turn to a large language model (LLM) for help. You start a conversation with a free conversational agent, but soon realize that the answers are not precise enough. Maybe the responses are too generic, or in some cases, they’re completely off the mark.

Then an idea hits you: wouldn't it be nice if you could incorporate your own information into the model?

Sure, one way to do it is through **fine-tuning**: all you need to do is prepare your data into a dataset, select a pretrained, open-weights model, set up the training and evaluation scripts, pay for the infrastructure and computing power, and finally—after a few days and several thousand dollars—you’ll have your fine-tuned model! Easy, right? What do you mean you don’t want to spend thousands of dollars fine-tuning a model?! Don’t you know AI is just for the rich?

Well, if you’re GPU-poor, you can try building a RAG system. It’s quite straightforward. First, take your knowledge base and embed it into a **vector database**. Then, when querying your LLM, simply add the top-k **most similar documents** of your knowledge base to the prompt, based on the input question. By doing this, you’re **injecting knowledge** that the LLM might not be aware of, helping it reason better and providing a more accurate answer.

I like to define a RAG system as a **component** in a conversational pipeline that extends the knowledge base of an LLM by using the prompt, all while leaving the model’s **weights unchanged**. As you can imagine, this technique is much cheaper and faster to prototype compared to model fine-tuning.

Of course, we don’t expect the full answer to a question to be contained within the retrieved documents. Instead, we assume that these documents will provide **relevant information** about the question, thereby aiding the agent’s reasoning process.


## Suturing the Code: Implementing RAG
Now that we have a solid understanding of what a RAG system is, let’s dive into some code that demonstrates this technique. It might seem a bit unconventional, but this time I’ll be using Python instead of CUDA! Remember, this is a relaxed article!

> 🚀 I started using [uv](https://astral.sh/blog/uv-unified-python-packaging) as my go-to Python package manager. I highly suggest you to try it, it's blazing fast!

### Storing the Memories
It may seem incredible, but a **vector database** is exactly what it sounds like—a database for vectors. Yes, I know, take a moment to let that sink in. What do you mean it was obvious? Well, I guess you're right. Anyhow, we’ll be using [Qdrant](https://qdrant.tech/), a blazing-fast vector database, written in Rust. It’s very easy to set up, thanks to their Docker image. Simply pull the image and make sure to install their Python client. For example:

```sh
uv add qdrant-client
``` 

The great thing about vector databases is that you can associate each vector with a **payload**, which is essentially a set of information about that data point. In our RAG case, we’ll map the text embedding to the text itself, allowing us to quickly find texts similar to the user’s prompt.

Since I’m not solving a specific problem here, I didn’t have any particular data to put in the vector database. So, I created a character named Mr. Fat Raccoon and generated some sentences about him. Here’s our example knowledge base:

```python
import pandas as pd
data = pd.DataFrame({
    'text': [
        "Mr. Fat Raccoon was born in Trash City.",
        "This raccoon, known as Mr. Fluffy, is 80cm long.",
        "Mr. Fat Raccoon weighs 27 kgs.",
        "In Trash City, a raccoon named Mr. Fat Raccoon was born.",
        "At 80cm long, Mr. Fat Raccoon is quite large.",
        "Weighing 27 kgs, Mr. Fat Raccoon is one hefty raccoon.",
        "Trash City is the birthplace of Mr. Fat Raccoon.",
        "Mr. Fat Raccoon, a native of Trash City, is known for his 80cm length.",
        "The weight of Mr. Fat Raccoon is 27 kgs.",
        "Born in Trash City, Mr. Fat Raccoon is 80cm long and weighs 27 kgs."
    ]
})
```

Now that we have the data, it’s time to populate the database. First, we instantiate the **embedding model**, which will be used to generate text embeddings. Then, we create a collection in Qdrant, where we will insert our data points.

When creating the collection, we must specify its **name**, the **vector size**—which corresponds to the size of the embeddings and is therefore model-dependent—and the type of **distance metric** we want to use. For this example, we’ll use cosine distance.

Each data point will be represented by its embedding and will include the original text in its payload. Here’s the code to do that:

```python
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

client = QdrantClient(url="http://localhost:6333")
collection_name = "raccoon_info"
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def create_knowledge_base():
    if collection_name not in client.get_collections().collections:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config= models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            )
        )
    embeddings = embedding_model.encode(data['text'].tolist())
    points = [
        models.PointStruct(id=idx, vector=embedding.tolist(), payload={"text": row["text"]})
        for (idx, row), embedding in zip(data.iterrows(), embeddings)
    ]

    client.upsert(
        collection_name=collection_name,
        points=points
    )
    print("Data inserted into Qdrant collection successfully.")
```

Great, we now have our knowledge base embedded in a vector database! We can proceede with our experiment.


### Asking about the unknown
Now, let’s choose an LLM and ask it about Mr. Fat Raccoon. What do you expect the output to be? I’m fairly confident the model will **hallucinate**, but let’s find out.

I decided to use Microsoft’s `Phi-3.5-mini-instruct`, just to ensure we’re working with a good conversational model.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-mini-instruct", trust_remote_code=True)
```

Here’s the function we’ll call to generate the response:

```python
def ask_question(prompt: str, inject_knowledge: bool = True) -> str:
    if inject_knowledge:
        prompt_vector = embedding_model.encode(prompt).tolist()  # type: ignore
        search_result = client.search(
            collection_name=collection_name,
            query_vector=prompt_vector,
            limit=2,
        )
        injection = "Considering that: \n" + "\n".join([point.payload["text"] for point in search_result])  # type: ignore
        prompt = injection + prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_length=128, do_sample=True, temperature=0.1)
    return tokenizer.decode(output[0], skip_special_tokens=True)
```
> ⚠️ Don't be an amateur! Be sure to always **use the same model** for both the embedding and retrieving phase!

Let’s give it a try:

```python
print(ask_question("What is double the weight of Mr. Fat Raccoon?", inject_knowledge=False))
```

The result is as follows:
> First, we need to calculate the weight of Mr. Fat Raccoon. We know that Mr. Fat Raccoon weighs 30 pounds more than Mr. Scary Raccoon. Since Mr. Scary Raccoon weighs 30 pounds, we add this to Mr. Fat Raccoon's weight. 
So, Mr. Fat Raccoon's weight = Mr. Scary Raccoon's weight + 30 pounds


Well, that’s disappointing—the model is just spitting out nonsense! First of all, Mr. Fat Raccoon weighs 27 kilos (we’re not confused Americans here; we use the metric system). And who on earth is Mr. Scary Raccoon?

Now, let’s see what happens when we add information to the prompt using RAG.


### Injecting the Insight: Retrieving Data for the Prompt
As you can see from the previous code snippet, when setting the `inject_knowledge` parameter to `True`, the pipeline changes slightly.
First, we use the embedding model to embed the input question. Then, we retrieve the top-k results from our vector database (in this case, the top two). The payload of the most similar data points is then **injected** into the user prompt.

Let’s see if things change when using RAG:
```python
print(ask_question("What is double the weight of Mr. Fat Raccoon?", inject_knowledge=True))
```
> To find double the weight of Mr. Fat Raccoon, we simply multiply his weight by 2:
27 kgs * 2 = 54 kgs
Double the weight of Mr. Fat Raccoon is 54 kgs.

That’s spot on! Well done, Phi-3.5!
Now, let’s check the prompt that generated this response:

> Considering that: 
The weight of Mr. Fat Raccoon is 27 kgs.
Mr. Fat Raccoon weighs 27 kgs.
What is double the weight of Mr. Fat Raccoon?

As you can see, our RAG system successfully inserted relevant information into the prompt, enabling the model to respond correctly. Easier said than done!


## Wrapping Up the Operation: Final Thoughts
Well, there you have it, folks! We’ve successfully implemented a RAG system, enhancing our LLM’s ability to provide **accurate responses** by injecting relevant information into the prompt. By leveraging a vector database like Qdrant, we avoided the costly and time-consuming process of fine-tuning, all while improving the model’s performance.

Remember, not every procedure requires a **complex** or **expensive** solution. Sometimes, a well-placed stitch—in this case, a bit of embedded knowledge—is all it takes to get the job done right. So next time you find yourself with a data-heavy problem, consider RAG as your go-to surgical tool.

Until next time, keep your scalpels sharp and your models smarter!