import numpy as np  
import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt
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



st.markdown ( 
'''
## Análisis Exploratorio de Datos

El propósito del análisis exploratorio es tener una idea completa de cómo son nuestros datos, antes de decidir qué técnica usar. 
Y como en la práctica los datos no son ideales, debemos organizarlos, entender su contenido, entender cuáles son las variables más relevantes y cómo se relacionan unas con otras, comenzar a ver algunos patrones, determinar qué hacer con los datos faltantes y con los datos atípicos, y finalmente extraer conclusiones acerca de todo este análisis. 
''' )

dl_movielens = DataLoader_Movielens ( )
dl_tmdb = DataLoader_TMDB ( )


merge_movilens = dl_movielens.get_merge_by_item_ids ( )
data_set = dl_movielens.data_set
user_set = dl_movielens.user_set
item_set = dl_movielens.item_set

st.markdown ( 
'''
### Análisis del Conjunto de Datos de Movielens
''' )

st.write ( merge_movilens )
  
# =======================================================================================================
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



# Raw Data
# Processed Data