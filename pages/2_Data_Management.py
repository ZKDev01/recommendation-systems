import numpy as np  
import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt
from typing import List, Dict, Any 
from surprise import Prediction, KNNBaseline, SVD, KNNWithMeans
from collections import defaultdict

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

@st.cache_resource()
def dataset ( generate_preprocessing: bool = True ) -> Dict[str,pd.DataFrame]:
  dl_movielens = DataLoader_Movielens ()
  dl_tmdb = DataLoader_TMDB ()

  if generate_preprocessing:
    dl_tmdb.preprocessing_set()

  tmdb_preprocessed = dl_tmdb.load_preprocessed()

  dict_dataset = { 
    'Movielens Rating Set' : dl_movielens.data_set,
    'Movielens User Set' : dl_movielens.user_set,
    'Movielens Item Set' : dl_movielens.item_set,
    'TMDB Preprocessed Set' : tmdb_preprocessed,
    'TMDB Movies Set' : dl_tmdb.movies_set
  }

  return dict_dataset

dict_dataset = dataset()

st.write (dict_dataset['TMDB Preprocessed Set'])



st.markdown (""" 
## Manejo y Análisis Exploratorio de los Conjuntos de Datos 

### MOVIELENS
""")

dl_movielens = DataLoader_Movielens ( )
st.write ("#### RANTING SET")
st.write (dl_movielens.data_set)
st.write ("#### ITEM SET")
st.write (dl_movielens.item_set)
st.write ("#### USER SET")
st.write (dl_movielens.user_set)
st.write ("#### MERGE BY ITEMS")
merge = dl_movielens.get_merge_by_item_ids()
st.write (merge)

#! Análisis de Generos
if st.button ('Análisis de géneros de las personas que calificaron'):
  labels = 'M', 'F'
  gender_counts = count_ratings_by_gender ( df=dl_movielens.user_set )
  sizes = ( gender_counts.iloc[0], gender_counts.iloc[1] )
  fig, ax = plt.subplots( )
  ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['blue', 'red'])
  st.pyplot ( fig )

#! Count-user-ratings
if st.button ('Función `count_user_ratings`'):
  st.write (count_user_ratings)

#! Get-top-K-movies
if st.button ('Función `get_top_K_movies`'):
  st.write (get_top_K_movies)

#! Count-people-by-age
if st.button ('Función `count_people_by_age`'):
  st.write (count_people_by_age)




st.markdown (""" 
### TMDB-5000 MOVIES
""")
dl_tmdb = DataLoader_TMDB ()
#dl_tmdb.preprocessing_set ()

#! Preprocessed-set 
if st.button ('Load-preprocessed-set'):
  st.write (dl_tmdb.load_preprocessed())

#! Count-Movies-by-Original-Language
if st.button ('Función `count_movies_by_original_language`'):
  st.write (count_movies_by_original_language)

#! top_K_movies_by_column

#! dl_tmdb.get_genders_list

#! get_movies_by_genders (preprocessed-set, genders)


