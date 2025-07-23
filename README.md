# NaiveRAG

## Introduction

A simple NaiveRAG pipeline designed for simple RAG use cases.

#### This is a work in progress Repo. This is an early implementation!

In the future abstract methods will be implemented as well as different chunking strategies, DBs, ...!

## Installation

First create a virtual environment. I recommend the usage of conda.

```
conda create -n naiverag python=3.12
conda activate naiverag
```

Then the project can be installed with:

```python
pip install -e .
```
Then create a .env file with your API Key specified:

```
API_KEY=...
```
Also be sure to look at the configurations in config.yaml! 