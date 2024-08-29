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



from src.Data_Management.data_loader import (
  DataLoader_Movielens,
  DataLoader_TMDB
)
from src.Data_Management.exploratory_data_analysis import (
  get_dataframe_from_data_set_group_by_rating,
)

from src.Recommendation_Model_Analysis.data_generator import DataGenerator
from src.Recommendation_Model_Analysis.metrics import Metrics
from src.Recommendation_Model_Analysis.model_factory import Model, HybridModel_Weighted
from src.Recommendation_Model_Analysis.utils import (
  get_top_n,

)





def testing_class ( ) -> None:
  """_summary_
  """

  st.write ( '# Clases del proyecto y aplicacion' )
  st.write ( '## Clase `DataLoader`' )

  data_loader = DataLoader_Movielens ( )
  st.write ( type( data_loader ) )
  
  st.write ( '## Test sobre Modelos' )

  df_user = data_loader.load_set ( 'DATA' )

  data_generator = DataGenerator (
    dataframe= df_user
  )
  data_generator.from_df_to_dataset()
  data_generator.train_test_split()

  model_knn_baseline = Model ( 
    model= KNNBaseline(),
    name= 'KNN Baseline' 
  ) 
  metrics = model_knn_baseline.evaluate ( data=data_generator )
  metrics.compute_metrics ( 'MAE', 'RMSE' )
  st.write ( model_knn_baseline )
  st.write ( metrics )
  
  model_svd = Model (
    model= SVD(),
    name= 'SVD'
  )

  metrics = model_svd.evaluate ( data=data_generator )
  metrics.compute_metrics ( 'MAE', 'RMSE' )
  st.write ( model_svd )
  st.write ( metrics )

  # Test Hybrid Model Class
  hybrid = HybridModel_Weighted (
    name= ' SVD x KNN with Means ', 
    models= [ 
      Model ( model=SVD(), name='SVD' ),
      Model ( model=KNNWithMeans( sim_options= { 'name': 'cosine', 'user_based': False } ), name='KNN with Means' ) ], 
    weights= [ 0.5, 0.5 ] )

  st.write ( hybrid )
  
  _, testset = data_generator.get_train_test_set ( )
  
  hybrid.fit ( data_generator=data_generator )
  predictions = hybrid.test ( testset )

  metrics = Metrics ( predictions=predictions )
  metrics.compute_metrics ( 'RMSE', 'MAE' )

  st.write ( metrics )

  top_n = get_top_n ( predictions=predictions, user_id=10, n=10 )
  st.write ( top_n )









def not_data_management ( ) -> None:

  st.write ( '# Dataset: TMDB 5000 Movies' )
  
  tmdb = DataLoader_TMDB ( )
  df = tmdb.get_preprocessed_set ( )

  # st.write ( tmdb.convert_preprocessed_set_to_list ( )[ 0 ] )
  st.write ( df )
      






def not_exploratory_data_analysis () -> None:
  """
  """

  dl_movielens = DataLoader_Movielens ()
  dl_movielens.load_set() 
  st.write ( '# Exploratory Data Analysis' ) 
  st.markdown(
    '''
    En esta seccion va a estar el Analisis Exploratorio de Datos
    '''
  )

  columns_configuration = {
    'userID': st.column_config.TextColumn(
      'UserID',
      help='ID del usuario',
      max_chars=100,
      width='medium'
    ),
    'itemID': st.column_config.TextColumn(
      'ItemID',
      help='ID del item',
      max_chars=100,
      width='medium'
    ),
    'rating': st.column_config.TextColumn(
      'Rating',
      help='Rating de la pelicula por el usuario',
      max_chars=100,
      width='medium'
    )
  }
  st.write ( '## Rating DataFrame' )
  loader = DataLoader_Movielens ( )
  df = loader.load_set ( 'DATA' )
  
  # different al original
  event = st.dataframe ( 
    df,
    column_config=columns_configuration,
    use_container_width=True,
    hide_index=True
  )

  st.write ( 'Analizar la distribucion de los ranking' )

  st.write ( '## See movie' )
  st.write ( 'Cuando el que use la aplicacion toque un ranking pueda mostrar las peliculas que la persona rankeo y cual es la pelicula, ademas de mostrar un analisis de la persona' )

  st.write ( '## Analisis de la informacion demografica de los usuarios')
  st.write ( 'Analisis de las edades, sacando el promedio, describe(), distribucion, imagen con un historigrama' )
  st.write ( 'Lo anterior pero con todos los aspectos de los usuarios, que sean utiles, por ejemplo, con localizacion no hace falta a nuestro entender')
  # tenemos en la informacion demografica un aspecto pais, si es asi usar un mapa de pais con una grafica de calor para analizar donde viven las personas que rankearon

  st.write ( '## Analisis de la informacion de las peliculas' )
  st.write ( 'Lo mismo que los usaurios pero con las peliculas' )
  st.write ( 'Analizar los generos de las peliculas' )
  st.write ( 'Top peliculas mejor rankeados' )
  st.write ( 'Top peliculas mejor rankeados por un filter, por ejemplo, misterio' )









def home ( ) -> None:
  """
  
  """
  
  st.markdown ( '''
  # Proyecto de Sistemas de Recomendación
  
  En este proyecto se investigó los diferentes modelos de implementación básicos. Además se presenta como último punto un sistema de recomendación usando como un Modelo Grande de Lenguaje (LLM o Large Language Model)
  
  ''')



def exploratory_data_analysis ( ) -> None:
  """ 
  
  """
  
  st.markdown ( 
  '''
  ## Análisis Exploratorio de Datos

  El propósito del análsiis exploratorio es tener una idea completa de cómo son nuestros datos, antes de decidir qué técnica usar. Y como en la práctica los datos no son ideales, debemos organizarlos, entender su contenido, entender cuáles son las variables más relevantes y cómo se relacionan unas con otras, comenzar a ver algunos patrones, determinar qué hacer con los datos faltantes y con los datos atípicos, y finalmente extraer conclusiones acerca de todo este análisis. 


  ''' )

  # load dataset 
  dl_movielens = DataLoader_Movielens ( )
  dl_tmdb = DataLoader_TMDB ( )
  
  # merge dataset of movielens 
  merge_movilens = dl_movielens.get_merge_by_item_ids ( )

  st.markdown ( 
  '''
  ### Conjunto de Datos de Movielens
  
  (descripcion del dataset)
  ''' )

  st.write ( merge_movilens )
  
  st.markdown ( 
  '''
  (analisis exploratorio de datos para el dataset de movielens)
  ''' )

  # preprocessed set of tmdb 
  preprocessed_set = dl_tmdb.get_preprocessed_set ( )
  
  st.markdown ( 
  '''
  ### Conjunto de Datos de TMDB 5000 Movies

  (descripcion del dataset)
  ''' )
  
  st.write ( preprocessed_set )


def recommendation_models ( ) -> None:
  """ 
  
  """
  
  pass

def llm_assistant ( ) -> None:
  """ 
  
  """
  
  st.markdown ( '''
  ## Sistema de Recomendación usando LLM
  
  ### Qué es un LLM?
  
  Un LLM es un modelo generativo ...
  ''')










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




