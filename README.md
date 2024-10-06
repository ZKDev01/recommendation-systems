# Recommendation Systems

## Project Description: Information Retrieval Systems
This project covers both fundamental and advanced concepts in the field of Recommendation Systems. This project encompasses:

- Traditional Recommendation Systems such as Collaborative Filtering-based Systems
- Advanced Recommendation Systems such as Hybrid Recommendation Systems and those using LLMs (Large Language Models) with the RAG (Retrieval-Augmented Generation) technique

This project integrates these knowledge areas, offering a small glimpse into the evolution of Recommendation Systems from their roots to modern innovations

## Requirements
Some of the main libraries used in our project are as follows:

### Surprise Library
Surprise is a library designed to implement recommendation systems based on collaborative filtering. Its name comes from "Simple Python Recommendation System Engine"

It offers a wide range of predefined collaborative filtering algorithms and tools for evaluating and comparing their performance. Additionally, it allows easy handling of both internal and custom datasets.

### LangChain Library
LangChain is a library to facilitate the development of artificial intelligence systems, especially those using model languages (LLMs).

LangChain focuses on providing tools and reusable components to create AI applications, particularly those involving conversations, content generation, and natural language processing.

## Setup Guide

1. Run the command `pip install -r requirements.txt`
2. In a `.env` file, enter the Google API key with `google_api_key`, like: `google_api_key='...'`
3. Run the command `streamlit run app.py`

To obtain the Google API key, please consult: https://aistudio.google.com

## Authors

- Raimel D. Romaguera Puig
- Manuel Alejandro Gamboa Hernández