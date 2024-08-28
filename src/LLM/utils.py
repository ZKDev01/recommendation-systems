import os
import dotenv

from typing import List

import numpy as np  
import pandas as pd

from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document



def load_environment () -> None:
  """
  Carga las variables de entorno necesarias para el funcionamiento del programa

  Especificamente buscar las variables de entorno siguientes:
  - `google_api_key` : clave de API de Google  
  """
  
  dotenv.load_dotenv()
  os.environ.setdefault ( 'google_api_key', os.getenv('google_api_key') )

def get_model() -> GoogleGenerativeAI:
  """
  Inicializa y devuelve una instancia de GoogleGenerativeAI
  con configuraciones predeterminadas 

  Contiene una funcion para cargar las credenciales de Google API 
  desde el entorno y crea una instancia de GoogleGenerativeAI
  usando el modelo 'gemini-1.5-pro'

  Args:

  Returns:
      GoogleGenerativeAI: instancia preconfigurada de GoogleGenerativeAI
  """

  load_environment()
  model = GoogleGenerativeAI(
    model='models/gemini-1.5-pro-latest',
    temperature=0.5
  )
  return model

def get_embedding() -> GoogleGenerativeAIEmbeddings:
  """
  Inicializa y devuelve una instancia de `GoogleGenerativeAIEmbeddings`

  Esta funcion carga las credenciales de Google API desde el entorno y crea una instancia de `GoogleGenerativeAIEmbeddings`

  Returns:
      GoogleGenerativeAIEmbeddings: instancia preconfigurada del embedding model 
  """

  load_environment()
  embedding = GoogleGenerativeAIEmbeddings(
    model='models/embedding-001'
  )
  return embedding 



def prompt_template_QA(question: str, k: int, model: GoogleGenerativeAI) -> str:
  """
  Este metodo construye un template de chat que incluye instrucciones claras para el modelo de IA sobre como responder 
  a una pregunta especifica y sugerir posibles preguntas relacionadas. 

  Utiliza parametro `k` para especificar la cantidad de recomendaciones de preguntas debe incluir en su respuesta

  Args:
      question (str): la pregunta especifica que se desea que el modelo responda 
      k (int): cantidad de recomendaciones de preguntas relacionadas que se deben incluir en la respuesta 
      model (GoogleGenerativeAI): instancia del modelo de IA utilizado para generar respuesta

  Returns:
      result (str): respuesta generada por el modelo, incluyendo tanto la respuesta directa a la pregunta como las recomendaciones de preguntas relacionadas
      
  """

  prompt = ChatPromptTemplate.from_template(
    """ 
    Se lo mas simple posible para responder la siguiente pregunta 
    y da algunas recomendaciones a preguntas que se parezcan al tema de la pregunta

    Solo devuelve la respuesta. Seguido de las preguntas. Ejemplo:
    
    Answer

    Posibles preguntas:
    - Pregunta sugerida 1
    - Pregunta sugerida 2
    - Pregunta sugerida 3  

    El numero de preguntas que sugieres debe estar fijado al siguiente numero:
    Numero de recomendaciones: {k}

    Q: {question}
    A: 
    """
  )
  
  chain = prompt | model 
  result = chain.invoke(
    {
      "question": question,
      "k": k
    })
  return result

