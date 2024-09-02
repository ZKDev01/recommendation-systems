# Sistemas de Recomendación

## Descripción del Proyecto de Sistemas de Recuperación de Información

Este proyecto aborda tanto conceptos fundamentales como avanzados en el campo de los Sistemas de Recomendación. Este proyecto comprende:
- Sistemas de Recomendación Tradicionales como Sistemas usando Filtrado Colaborativo
- Sistemas de Recomendación Avanzados como los Sistemas de Recomendación Híbridos y Sistemas de Recomendación utilizando LLM (Large Language Models) con la técnica RAG (Retrieval-Augmented Generation)

Este proyecto integra estos conocimientos, ofreciendo una pequeña visión de la evolución de los Sistemas de Recomendación desde sus raíces hasta innovaciones modernas  

## Requerimientos 

Algunas de las librerías principales que utiliza nuestro proyecto son las siguientes:

### Librería Surprise

Surprise es una librería diseñada para implementar sistemas de recomendación basados en filtrado colaborativo. Su nombre proviene de "Simple Python Recommendation System Engine" 

Esta ofrece una amplia gama de algoritmos de filtrado colaborativo predefinidos y herramientas para evaluar y comparar su rendimiento. Además de permitir manejar fácilmente conjuntos de datos tanto internos como personalizados. 

### Librería LangChain

LangChain es una librería para facilitar el desarrollo de sistemas de inteligencia artificial, especialmente aquellos que utilizan lenguajes de modelo (LLMs). 

LangChain se centra en proporcionar herramientas y componentes reutilizables para crear aplicaciones de IA, particularmente aquellas que involucran conversaciones, generación de contenido y procesamiento de lenguaje natural

## Setup Guide

1. Ejecutar el comando `pip install -r requirements.txt`
2. En un archivo `.env` introducir la api key de Google con `google_api_key`, como: `google_api_key='...'`
3. Ejecutar el comando `streamlit run app.py` 

Para conseguir la api key de google puedes consultar: https://aistudio.google.com

