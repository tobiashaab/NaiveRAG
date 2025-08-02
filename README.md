# NaiveRAG

![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-beta-yellow.svg)
![Last Commit](https://img.shields.io/github/last-commit/tobiashaab/NaiveRAG)
![Contributors](https://img.shields.io/github/contributors/tobiashaab/NaiveRAG)

---

## 🚀 Introduction

**NaiveRAG** is a simple Retrieval-Augmented Generation (RAG) pipeline created for education purposes.

Future plans include:
- Support for different strategies - such as recursive token splitting.
- CLI args and a runnable bash script.


## 📦 Installation

### 1. Create a Virtual Environment

Using **conda** is recommended:

```bash
conda create -n naiverag python=3.12
conda activate naiverag
```

### 2. Install the Project

Install in editable mode:

```bash
pip install -e .
```

### 3. Set Environment Variables

Create a `.env` file in the project root:

```env
API_KEY=...
```

### 4. Configure the Pipeline

Adjust the settings in [`config.yaml`](./config.yaml)!

### 5. Run the Pipeline
After setting everything up, you can run the pipeline using the following command:

```bash
python run.py
```

## 📚 Documentation

### 1. Project Structure


The actual RAG pipeline is located in the [`/rag_pipeline`](./rag_pipeline) directory, which contains the following subdirectories:

- [`api/`](./rag_pipeline/api) – Implements API endpoints for language and embedding model APIs  
- [`chunking/`](./rag_pipeline/chunking) – Contains strategies for chunking documents  
- [`db/`](./rag_pipeline/db) – Contains vector store implementations  

Each of these directories includes a [`base.py`](./rag_pipeline/api/base.py) file that defines an abstract base class. These classes are then implemented in specific files (i.e. [`gemini.py`](./rag_pipeline/api/gemini.py) for the Gemini API).

The correct implementation is automatically selected based on your settings in [`config.yaml`](./config.yaml).

Utility functions (i.e., for loading the config) are located in [`/util`](./util).
