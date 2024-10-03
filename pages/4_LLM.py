from collections import defaultdict

import streamlit as st
import pandas as pd 
import numpy as np  

from surprise import ( 
  Prediction,
  KNNBaseline,
  SVD,
  KNNWithMeans
)

from src.data_management.utils import *
from src.data_management.data_loader import *
from src.data_management.exploratory_data_analysis import *

from src.llm_components.utils import *
from src.llm_components.vectorstore import *
from src.llm_components.chat_history import *

from src.recsys_analysis.utils import *
from src.recsys_analysis.metrics import *
from src.recsys_analysis.model_factory import *
from src.recsys_analysis.data_generator import *



def llm_assistant ( ) -> None:
  """ 
  Función principal para implementar un asistente de recomendación de películas utilizando un LLM y 
  búsqueda de similitud vectorial. 

  Esta función realiza las siguientes tareas:

  1. Carga y preprocesa el conjunto de datos de películas de TMDB.
  2. Crea bases de datos vectoriales personalizadas para cada género de película 
  3. Inicializa un chat con historial utilizando un LLM
  4. Implementa una interfaz de usuario en Streamlit para interactuar con el asistente
  5. Procesa las consultas del usuario, realiza búsquedas de similitud y genera recomendaciones personalizadas 

  Args:
      Ninguno
  
  Returns: 
      None
  
  Raises: 
      No se anticipan excepciones específicas, pero puede lanzar errores
  """

  # cargar tmdb-dataset
  dl_tmdb = DataLoader_TMDB ( )
  
  # obtener el conjunto de datos preprocesado 
  preprocessed_set = dl_tmdb.get_preprocessed_set ( )
  
  # obtener la lista de generos para filtrar con el conjunto de datos preprocesado 
  genders = dl_tmdb.get_genders_list ( )
  
  # obtenemos las peliculas separadas por generos 
  results = count_movies_by_genders_list ( preprocessed_set, genders )
  movies: dict = get_movies_by_genders ( preprocessed_set, genders )
  
  genders = genders[ 0:5 ]

  # obtenemos por cada tupla del dataframe una cadena de texto
  for i in genders:
    movies[ i ] = dl_tmdb.convert_preprocessed_set_to_list( dataframe=movies[ i ] )

  # aqui estaran almacenadas las bases de datos vectoriales (cada una correspondera a un genero) 
  vectorstore : list[ Faiss_Vectorstore ] = [ ]

  for i in genders:
    if results[i] <= 100:
      vectorstore.append ( Faiss_Vectorstore ( 
        movies=movies[ i ], 
        description=i
      ) )
    else:
      vectorstore.append ( Faiss_Vectorstore (
        movies=movies[ i ][ 0:100 ],
        description=i
      ) )

  # finalmente tenemos un objeto que contiene todas las bases de datos vectoriales
  personalized_vs = Personalized_Vectorstores ( vectorstore )  
  # y otro que usa el LLM para generar las respuesta/recomendaciones y con la característica de chat-history
  llm_with_history_chat = ChatHistory (  )
  llm_with_history_chat.make_chain()

  st.write ( '## RecSys using RAG' )
  # Initialize chat history
  if "messages" not in st.session_state:
    st.session_state.messages = []

  # Display chat messages from history on app rerun
  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])

  # React to user input
  if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # MAGIC HERE
    results_of_similarity_search = personalized_vs.similarity_search ( prompt, 5 )
    
    prompt = llm_with_history_chat.to_answer_query ( query=prompt, movies=results_of_similarity_search )

    response = f"IA: {prompt}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
      st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


llm_assistant()