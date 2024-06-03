## Prerequisites for running the solution:

Clone the repo:
```console
git clone https://github.com/shivachittamuru/Financial-Research-Assistant.git
cd Financial-Research-Assistant
```

Create a conda environment:
```console
conda create --name assistant python=3.10
conda activate assistant
```

Install `PIP` requirements
```console
pip install -r requirements.txt
```

Update the values in `.env-sample` and rename it to `.env`.


## Polygon.ai API

Go to [Polygon Website](https://polygon.io/), and create a free API key. Add the key in .env file


## LangGraph

LangGraph is a library that enables the creation of stateful applications involving multiple actors (like users, agents, or characters) interacting with Large Language Models (LLMs). It allows developers to define actors, attributes, relationships, and behaviors using a graph-based representation. It’s an extension of the LangChain library for coordinating multiple chains or actors in a cyclic computational process.

Get started with this [intro tutorial](https://langchain-ai.github.io/langgraph/tutorials/introduction/)







