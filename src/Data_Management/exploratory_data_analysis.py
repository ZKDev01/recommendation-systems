import os

from typing import Any

import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_distances


# Funciones utiles para el analisis exploratorio de datos en el dataset: MOVIELENS

def calculate_sim_matrix ( df: pd.DataFrame ) -> np.ndarray:
  """_summary_

  Args:
      df (pd.DataFrame): _description_

  Returns:
      np.ndarray: _description_
  """

  # 1. Pivotar el DataFrame para facilitar el calculo de la matriz de similitud
  pivot_df = df.pivot ( index='userID', columns='itemID', values='rating' ).fillna ( 0 )

  # 2. Normalizar los datos para que cada usuario tenga un promedio de calificacion
  normalized_pivot_df = pivot_df.div ( pivot_df.sum( axis=1 ), axis=0 )

  # 3. Convertir el DataFrame normalizado a una matriz de NumPy
  user_profiles = normalized_pivot_df.values

  # 4. Calcular la matriz de similitud coseno
  sim_matrix = cosine_distances ( user_profiles )
  return sim_matrix 

def get_dataframe_from_data_set_group_by_rating ( df: pd.DataFrame, head: int = 10, ascending: bool = True ) -> pd.DataFrame:
  """_summary_

  Args:
      df (pd.DataFrame): _description_
      head (int, optional): _description_. Defaults to 10.
      ascending (bool, optional): _description_. Defaults to True.

  Returns:
      pd.DataFrame: _description_
  """
  grouped = df.groupby ( 'userID' )['rating'].count().reset_index ( name='rating' )
  grouped_sorted = grouped.sort_values ( by='rating' )
  return grouped_sorted.head ( head, ascending=ascending )

def get_gender_counts_from_data_set ( df: pd.DataFrame ) -> pd.Series[ int ]:
  """_summary_

  Args:
      df (pd.DataFrame): _description_

  Returns:
      pd.Series[ int ]: _description_
  """
  return df [ 'gender' ].value_counts ( )



def statistic_about_ratings ( ) :
  pass




# Funciones utiles para el analisis exploratorio de datos en el dataset: TMDB 5000

# cambiar el nombre
def get_counts_of_movies_by_original_language ( df_credit: pd.DataFrame, df_movies: pd.DataFrame ) -> pd.Series[int]:
  """_summary_

  Args:
      df_credit (pd.DataFrame): _description_
      df_movies (pd.DataFrame): _description_

  Returns:
      pd.Series[int]: _description_
  """

  df_merge = df_movies.merge ( df_credit, on='title' )
  return df_merge [ 'original_language' ].value_counts( )

def get_top_K_of_movies ( merge: pd.DataFrame, columns: list[str] = [ 'title', 'budget', 'revenue' ], column: str = 'budget', K: int = 10, ascending: bool = False ) -> pd.DataFrame:
  """_summary_

  Args:
      merge (pd.DataFrame): _description_
      columns (list[str], optional): _description_. Defaults to [ 'title', 'budget', 'revenue' ].
      column (str, optional): _description_. Defaults to 'budget'.
      K (int, optional): _description_. Defaults to 10.
      ascending (bool, optional): _description_. Defaults to False.

  Returns:
      pd.DataFrame: _description_
  """

  result = merge[ columns ].sort_values ( column, ascending=ascending )
  return result.head( K )

