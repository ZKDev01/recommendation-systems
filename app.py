import streamlit as st
import pandas as pd 
import numpy as np  

from src.data_loader import DataLoader



def testing_class ( ) -> None:
  st.write ( '# Testing class' )





def intro () -> None: 
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
  loader = DataLoader ( )
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
    'MAIN': intro,
    'EDA': exploratory_data_analysis,
    'User-based CF': user_based_cf,
    'Item-based CF': None,
    'Content-based F': None,
    'ML based F: Baseline': None
  }
  
  deploy = st.sidebar.selectbox ( 'Choose:', page_names_to_funcs.keys() )
  page_names_to_funcs [ deploy ]()

if __name__ == '__main__':
  main()






