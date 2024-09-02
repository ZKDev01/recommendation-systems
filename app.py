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

import matplotlib.pyplot as plt


from src.Data_Management.data_loader import (
  DataLoader_Movielens,
  DataLoader_TMDB
)
from src.Data_Management.exploratory_data_analysis import (
  count_ratings_by_gender,
  count_user_ratings,
  get_top_K_movies,
  count_people_by_age,

  count_movies_by_original_language, 
  top_K_movies_by_column,
  count_movies_by_genders_list,
  get_movies_by_genders
)

from src.Recommendation_Model_Analysis.data_generator import DataGenerator
from src.Recommendation_Model_Analysis.metrics import Metrics
from src.Recommendation_Model_Analysis.model_factory import (
  Model, 
  HybridModel_Weighted
)
from src.Recommendation_Model_Analysis.utils import (
  get_top_n,
  get_evaluation_and_comparison_of_machine_learning_models
)


from src.LLM.vectorstore import (
  Faiss_Vectorstore,
  Personalized_Vectorstores
)
from src.LLM.chat_history import (
  ChatHistory
)




def home ( ) -> None:
  """
  Esta función se encarga de crear y mostrar el contenido de la página principal del proyecto.
  Presenta una descripción general del proyecto.

  Args:
      Ninguno

  Returns:
      None

  Raises:
      No se anticipan excepciones específicas, pero puede lanzar errores si falla la carga de la imagen.

  Notas:
      - Utiliza Streamlit para crear el contenido de la página.
      - Muestra un título y una descripción del proyecto.
      - Incluye una imagen generada por Canva.
  """

  st.markdown ( '''
  # Proyecto de Sistemas de Recomendación 
  
  En este proyecto se investigó los diferentes modelos de implementación básicos. 
  Además se presenta como último punto un sistema de recomendación usando un Modelo Grande de Lenguaje (LLM o Large Language Model) junto con una técnica mejorada para la recuperación de información (RAG)
  ''')

  st.image("assets/image-1.png", caption="Foto generada por Canva")




def exploratory_data_analysis ( ) -> None:
  """ 
  Esta función se encarga de cargar los conjuntos de datos de MovieLens y TMDB 5000 Movies,
  realizar diferentes análisis estadísticos y algunas visualizaciones para entender mejor la estructura
  y distribución de los datos.

  Args: 
      Ninguno

  Returns:
      None
  
  Raises:
      No se anticipan excepciones específicas, pero puede lanzar errores si falla la carga de los datos.
  """
  
  st.markdown ( 
  '''
  ## Análisis Exploratorio de Datos

  El propósito del análisis exploratorio es tener una idea completa de cómo son nuestros datos, antes de decidir qué técnica usar. 
  Y como en la práctica los datos no son ideales, debemos organizarlos, entender su contenido, entender cuáles son las variables más relevantes y cómo se relacionan unas con otras, comenzar a ver algunos patrones, determinar qué hacer con los datos faltantes y con los datos atípicos, y finalmente extraer conclusiones acerca de todo este análisis. 
  ''' )

  # load dataset 
  dl_movielens = DataLoader_Movielens ( )
  dl_tmdb = DataLoader_TMDB ( )
  
  # merge dataset of movielens and other dataset
  merge_movilens = dl_movielens.get_merge_by_item_ids ( )
  data_set = dl_movielens.data_set
  user_set = dl_movielens.user_set
  item_set = dl_movielens.item_set

  st.markdown ( 
  '''
  ### Análisis del Conjunto de Datos de Movielens
  
  ''' )

  st.write ( merge_movilens )
  
  # ========================================================================================================
  st.markdown ( 
  '''
  Grafica de Géneros de las personas que calificaron
  ''' )
  gender_counts = count_ratings_by_gender ( df=user_set )

  labels = 'M', 'F'
  sizes = ( gender_counts.iloc[0], gender_counts.iloc[1] )

  fig, ax = plt.subplots( )
  ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['blue', 'red'])

  st.pyplot ( fig )

  # default values
  grouped = count_user_ratings ( df=data_set, head=10, ascending=True )
  st.write ( grouped )

  # ascending = False
  grouped = count_user_ratings ( df=data_set, head=10, ascending=False )
  st.write ( grouped )

  # ascending = True, but head = -1 (return fullset)
  grouped = count_user_ratings ( df=data_set, head=-1, ascending=True )
  st.write ( grouped )
  
  grouped = get_top_K_movies ( df=data_set )
  st.write ( grouped )

  grouped = count_people_by_age ( df=user_set )
  st.write ( grouped )
  
  preprocessed_set = dl_tmdb.get_preprocessed_set ( )
  movies_set = dl_tmdb.get_movies_dataset ( )
  credit_set = dl_tmdb.get_credit_dataset ( )
  
  merge = movies_set.merge ( credit_set, on='title' )

  st.markdown ( 
  '''
  ### Análisis del Conjunto de Datos de TMDB 5000 Movies
  ''' )
  
  st.write ( preprocessed_set )

  results = count_movies_by_original_language ( merge ).head ( 15 )
  st.write ( results )

  results = top_K_movies_by_column ( merge, column='budget', ascending=False, K=10 )
  st.write ( results )

  results = top_K_movies_by_column ( merge, column='revenue', ascending=False, K=10 )
  st.write ( results )
  
  genders = dl_tmdb.get_genders_list ( )
  st.write ( genders )

  results = count_movies_by_genders_list ( preprocessed_set, genders )
  st.write ( results )

  results = get_movies_by_genders ( preprocessed_set, genders )
  for i in genders:
    st.write ( i )
    st.write ( results [ i ] )



def recommendation_models ( ) -> None:
  """ 
  
  """
  
  st.markdown ( 
  '''
  ## Modelos de Recomendación

  A continuación se presentan comparaciones y análisis de los modelos tradicionales de sistemas de recomendación usando como conjunto de datos para en entrenamiento y la evaluación Movielens  

  '''
  )

  dl_movielens = DataLoader_Movielens ( )
  data_set = dl_movielens.data_set

  data_generator = DataGenerator ( 
    dataframe=data_set
  )
  data_generator.from_df_to_dataset()
  data_generator.train_test_split()

  st.markdown (  
  '''
  ### Comparación y Evaluación de los Diferentes Modelos provistos por Surprise
  ''')  
  results = get_evaluation_and_comparison_of_machine_learning_models( data_set )

  st.write ( results )
  
  st.markdown (  
  '''
  Analizando los modelos KNN Baseline y SVD 'clasico'
  ''')

  model_knn_baseline = Model ( 
    model=KNNBaseline(),
    name='KNN Baseline'
  )
  metrics = model_knn_baseline.evaluate ( data=data_generator )
  metrics.compute_metrics ( 'MAE', 'RMSE' )

  st.write ( '===============================' )
  st.write ( model_knn_baseline )
  st.write ( metrics )
  
  model_svd = Model (
    model=SVD(),
    name='SVD'
  )
  metrics = model_svd.evaluate ( data=data_generator )
  metrics.compute_metrics ( 'MAE', 'RMSE' )

  st.write ( '===============================' )
  st.write ( model_svd )
  st.write ( metrics )

  st.markdown (  
  '''
  ### Sistema de Recomendación Híbrido  
  '''
  )

  hybrid_model = HybridModel_Weighted (
    name='SVD x KNN with Means',
    models=[
      Model(
        model=SVD(),
        name='SVD'
      ),
      Model(
        model=KNNWithMeans ( sim_options= { 'name': 'cosine', 'user_based': False } ),
        name='KNN with Means'
      )
    ],
    weights=[0.5,0.5]
  )

  st.write ( hybrid_model )
  
  _, testset = data_generator.get_train_test_set ( )
  hybrid_model.fit ( data_generator )
  predictions = hybrid_model.test ( testset )

  metrics = Metrics ( predictions=predictions )
  metrics.compute_metrics ( 'RMSE', 'MAE' )

  st.write ( metrics )

  # PEDIR AL USUARIO QUE INSERTE UN USER_ID 
  top_n = get_top_n ( predictions=predictions, user_id=10, n=10 )
  st.write ( top_n )





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




def main () -> None:
  """
    Función principal del programa que configura y ejecuta la interfaz de usuario.

    Esta función crea un menú lateral con opciones para navegar entre diferentes secciones
    del proyecto y ejecuta la función correspondiente según la selección del usuario.

    Args:
        Ninguno

    Returns:
        None

    Raises:
        No se anticipan excepciones específicas, pero puede lanzar errores si las funciones
        asociadas a cada opción fallan durante su ejecución.

    Notas:
        - Crea un diccionario que mapea nombres de páginas con sus funciones correspondientes.
        - Utiliza `st.sidebar.selectbox` para crear un menú desplegable en el sidebar de Streamlit.
        - Ejecuta la función seleccionada por el usuario.

    Ejemplo:
        Si el usuario selecciona 'Análisis Exploratorio de Datos', se ejecutará la función `exploratory_data_analysis()`.
    """
  
  page_names_to_funcs = {
    'Inicio': home,
    'Análisis Exploratorio de Datos': exploratory_data_analysis,
    'Modelos de Recomendación': recommendation_models,
    'Modelos de Recomendación usando Modelos de Lenguaje': llm_assistant,
  }
  
  deploy = st.sidebar.selectbox ( 'Seleccione:', page_names_to_funcs.keys(), disabled=False )
  page_names_to_funcs [ deploy ]()



if __name__ == '__main__':

  main ( )




