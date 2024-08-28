from typing import Any

import pandas as pd 
import matplotlib

# Funciones utiles para el analisis exploratorio de datos en el dataset: MOVIELENS

def statistic_about_ratings ( ) :
  pass




# Funciones utiles para el analisis exploratorio de datos en el dataset: TMDB 5000

# cambiar el nombre
def get_counts_of_movies_by_original_language ( df_credit: pd.DataFrame, df_movies: pd.DataFrame ) -> pd.Series[int]:
  df_merge = df_movies.merge ( df_credit, on='title' )
  return df_merge [ 'original_language' ].value_counts( )

