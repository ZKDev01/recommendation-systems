import os

from typing import Any

import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_distances



# Funciones utiles para el analisis exploratorio de datos en el dataset: MOVIELENS

def count_ratings_by_gender ( df: pd.DataFrame ) -> pd.Series:
  """
  Esta función calcula el número de personas que calificaron items, agrupados por su género

  Args:
      df (pd.DataFrame) -> Un DataFrame que contiene datos de calificaciones. Debe tener una columna `gender` que representa el género de cada persona

  Returns:
      pd.Series -> Una Serie donde el índice representa géneros y los valores representan el número de personas de cada género que realizaron calificaciones
  """
  return df [ 'gender' ].value_counts ( )

def count_user_ratings ( df: pd.DataFrame, head: int = 10, ascending: bool = True ) -> pd.DataFrame:
  """
  Esta función agrupa los datos por `userID` y cuenta el número total de valoraciones realizadas por cada usuario. Los resultados pueden ser ordenados de manera ascendente o descendente según sea necesario.
  
  Args:
      df (pd.DataFrame) ->  Un DataFrame que contiene datos de valoraciones. Debe tener las columnas `userID` y `rating`
      head (int, optional) -> Número de usuarios a mostrar en los resultados. Por defecto es 10. Si se establece en -1, se muestran todos los usuarios.
      ascending (bool, optional) -> Indica si ordenar los resultados de manera ascendente. Por defecto es True (orden ascendente).

  Returns:
      pd.DataFrame -> Un DataFrame con dos columnas: `userID` (ID del usuario) y `rating` (número total de valoraciones realizadas por el usuario)
  """

  grouped = df.groupby ( 'userID' )['rating'].count().reset_index ( name='rating' )
  grouped_sorted = grouped.sort_values ( by='rating', ascending=ascending )
  
  return grouped_sorted if head == -1 else grouped_sorted.head( head )

def get_top_K_movies ( df: pd.DataFrame, K: int = 10 ) -> pd.DataFrame:
  """
  Esta función calcula el promedio de rating para cada película en el DataFrame, ordena los resultados de mayor a menor promedio y devuelve las K películas mejor puntuadas

  Args:
      df (pd.DataFrame) -> Un DataFrame que contiene datos de calificaciones. Debe tener las columnas `itemID` y `rating`.
      K (int, optional) -> Número de películas a retornar. Por defecto es 10.

  Returns:
      pd.DataFrame -> Un DataFrame con dos columnas: `itemID` (ID de la película) y `mean_rating` (el promedio de rating para esa película)
  """
  
  # Agrupar por item_id y calcualr el promedio de rating
  avg_ratings = df.groupby ( 'itemID' )['rating'].mean().reset_index()

  # renombrar las columnas 
  avg_ratings.columns = [ 'itemID', 'mean_rating' ]

  # ordenar de mayor a menor rating y tomar los K mejores
  top_k = avg_ratings.sort_values( 'mean_rating', ascending=False ).head( K )

  return top_k

def count_people_by_age ( df: pd.DataFrame ) -> pd.Series:
  """
  Esta función realiza un conteo de la distribución de edades en el DataFrame dado.

  Args:
      df (pd.DataFrame): Un DataFrame que contiene datos de usuario. Debe tener una columna `age` representando la edad de cada persona.

  Returns:
      pd.DataFrame: Una Serie donde el índice representa las edades y los valores representan el índice de personas de cada edad.
  """
  return df [ 'age' ].value_counts ( )






# Funciones utiles para el analisis exploratorio de datos en el dataset: TMDB 5000

# get_counts_of_movies_by_original_language
def count_movies_by_original_language ( merge: pd.DataFrame ) -> pd.Series:
  """
  Cuenta el número de películas por cada idioma original en el DataFrame

  Args:
      merge (pd.DataFrame) -> El DataFrame que contiene información sobre las películas, que debe incluir una columna llamada `original_language`

  Returns:
      pd.Series -> Una Serie con los nombres de los idiomas originales como índices y sus respectivas cuentas como valores.
  """
  return merge [ 'original_language' ].value_counts ( )

def top_K_movies_by_column ( merge: pd.DataFrame, column: str, ascending: bool = False, K: int = 10 ) -> pd.DataFrame:
  """
  Obtiene los primeros K películas por una columna específica

  Args:
      merge (pd.DataFrame) -> El DataFrame que contiene información sobre las películas
      column (str) -> El nombre de la columna por la cual se quiere ordenar
      ascending (bool, optional) -> Si el valor es True, el resultado estará ordenado en orden ascendente, de lo contrario, descendente. Por defecto es False.
      K (int, optional) -> Número de películas que se deben retornar. Si es -1, se regresan todas las películas. Por defecto es 10.

  Returns:
      pd.DataFrame -> Un DataFrame con las primeras K películas ordenadas según la columna especificada. 
  """
  results = merge[ [ 'title', 'budget', 'revenue' ] ].sort_values ( column, ascending=ascending )
  return results if K == -1 else results.head( K )

def get_movies_by_genders ( df: pd.DataFrame, genders: str ) -> dict:
  result = { }
  for gender in genders:
    result[ gender ] = df[ df['genders'].apply ( lambda x: gender in x ) ] 
  return result

def count_movies_by_genders ( merge: pd.DataFrame, target: str ) -> int:
  """_summary_

  Args:
      merge (pd.DataFrame): _description_

  Returns:
      pd.Series: _description_
  """
  def contains_gender ( genders_list: list[str] ):
    return target in genders_list

  return merge [ 'genders' ].apply( contains_gender ).sum( )

def count_movies_by_genders_list ( merge: pd.DataFrame, genders: list[str] ) -> dict[ str, int ]:
  result = { }
  for gender in genders:
    result[ gender ] = count_movies_by_genders ( merge, gender )
  return result

