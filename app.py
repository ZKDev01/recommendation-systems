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



from src.Data_Management.data_loader import DataLoader_Movielens

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













def intro () -> None: 
  """Presentacion del proyecto
  """


  st.write ( '# Recommendation Systems' )

  st.markdown(
    '''
    Explicacion de los sistemas de recomendacion:
    
    - User-based collaborative filtering
    - Item-based collaborative filtering -> Weighted Slope One
    - Content-based filtering
    - Machine Learning based filtering -> Baseline
    '''
  )





def exploratory_data_analysis () -> None:
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






def user_based_cf () -> None:
  st.write ( '# User-based Collaborative Filtering' )








def main () -> None:
  page_names_to_funcs = {
    'Principal': intro,
    'Analisis Exploratorio de Datos': exploratory_data_analysis,
    'Filtrado Colaborativo Basado en Usuarios': user_based_cf,
    'Filtrado Colaborativo Basado en Items': None,
    'Filtrado Basado en Contenido': None,
    'ML based F: Baseline': None
  }
  
  deploy = st.sidebar.selectbox ( 'Choose:', page_names_to_funcs.keys() )
  page_names_to_funcs [ deploy ]()

if __name__ == '__main__':
  main()

  # testing_class ( )







