from typing import List

from langchain.embeddings.gpt4all import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

embeddings = GPT4AllEmbeddings() # this downloads an embedding model if not cached


def get_text_chunks(text):
    # create chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_text(text)
    return docs


def generate_index(chunks_list: List) -> FAISS:
    # creating vectorstore with indexed file
    return FAISS.from_texts(chunks_list, embeddings)


def save_index(text, index_path):
    # saving index files to project dirs
    sources = get_text_chunks(text)
    vectorstore = generate_index(sources)
    vectorstore.save_local(index_path)


def ingest_text(text, content_name, role):
    valid_roles = ["EMPLOYEE", "USER"]
    if role in valid_roles:
        save_index(text, f"indexes/{role}/{content_name}")
    else:
        raise TypeError("Role does not exist")


if __name__ == '__main__':
    # langchain's github readme
    ingest_text("""LangChain
     Building applications with LLMs through composability 

Release Notes lint test Downloads License: MIT Twitter  Open in Dev Containers Open in GitHub Codespaces GitHub star chart Dependency Status Open Issues

Looking for the JS/TS version? Check out LangChain.js.

Production Support: As you move your LangChains into production, we'd love to offer more comprehensive support. Please fill out this form and we'll set up a dedicated support Slack channel.

Quick Install
pip install langchain or pip install langsmith && conda install langchain -c conda-forge

 What is this?
Large language models (LLMs) are emerging as a transformative technology, enabling developers to build applications that they previously could not. However, using these LLMs in isolation is often insufficient for creating a truly powerful app - the real power comes when you can combine them with other sources of computation or knowledge.

This library aims to assist in the development of those types of applications. Common examples of these applications include:

Question Answering over specific documents

Documentation
End-to-end Example: Question Answering over Notion Database
 Chatbots

Documentation
End-to-end Example: Chat-LangChain
 Agents

Documentation
End-to-end Example: GPT+WolframAlpha
 Documentation
Please see here for full documentation on:

Getting started (installation, setting up the environment, simple examples)
How-To examples (demos, integrations, helper functions)
Reference (full API docs)
Resources (high-level explanation of core concepts)
 What can this help with?
There are six main areas that LangChain is designed to help with. These are, in increasing order of complexity:

 LLMs and Prompts:

This includes prompt management, prompt optimization, a generic interface for all LLMs, and common utilities for working with LLMs.

 Chains:

Chains go beyond a single LLM call and involve sequences of calls (whether to an LLM or a different utility). LangChain provides a standard interface for chains, lots of integrations with other tools, and end-to-end chains for common applications.

 Data Augmented Generation:

Data Augmented Generation involves specific types of chains that first interact with an external data source to fetch data for use in the generation step. Examples include summarization of long pieces of text and question/answering over specific data sources.

 Agents:

Agents involve an LLM making decisions about which Actions to take, taking that Action, seeing an Observation, and repeating that until done. LangChain provides a standard interface for agents, a selection of agents to choose from, and examples of end-to-end agents.

 Memory:

Memory refers to persisting state between calls of a chain/agent. LangChain provides a standard interface for memory, a collection of memory implementations, and examples of chains/agents that use memory.

 Evaluation:

[BETA] Generative models are notoriously hard to evaluate with traditional metrics. One new way of evaluating them is using language models themselves to do the evaluation. LangChain provides some prompts/chains for assisting in this.

For more information on these concepts, please see our full documentation.

 Contributing
As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see here.""", "langchain", "EMPLOYEE")
