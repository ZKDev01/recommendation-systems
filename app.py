import streamlit as st
import pandas as pd 
import numpy as np  

from src.load_dataset import (
  load_specific_set
)

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

def eda () -> None:
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
  df = load_specific_set ( 'RATING' )

  # mmm
  df.drop ( 'timestamp', axis=1, inplace=True )

  # different al original
  event = st.dataframe ( 
    df,
    column_config=columns_configuration,
    use_container_width=True,
    hide_index=True
  )




def user_based_cf () -> None:
  st.write ( '# User-based Collaborative Filtering' )



def main () -> None:
  page_names_to_funcs = {
    'MAIN': intro,
    'EDA': eda,
    'User-based CF': user_based_cf,
    'Item-based CF': None,
    'Content-based F': None,
    'ML based F: Baseline': None
  }
  
  deploy = st.sidebar.selectbox ( 'Choose:', page_names_to_funcs.keys() )
  page_names_to_funcs [ deploy ]()

if __name__ == '__main__':
  main()






