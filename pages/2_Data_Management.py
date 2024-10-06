import numpy as np  
import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt
from typing import List, Dict, Any 

from src.data_management.utils import *
from src.data_management.data_loader_tmdb import *
from src.data_management.data_loader_movielens import *


st.set_page_config(
  page_title='Data Management',
  layout='wide'
)


@st.cache_resource()
def load_dataset ( generate_preprocessing:bool=False ) -> Dict[str,pd.DataFrame]:
  dl_movielens = DataLoader_Movielens ()
  dl_tmdb = DataLoader_TMDB ()

  if generate_preprocessing:
    dl_tmdb.preprocessing_set()

  movielens_merge = dl_movielens.get_merge_by_item_ids()
  tmdb_preprocessed = dl_tmdb.load_preprocessed()
  tmdb_tags_set =  dl_tmdb.create_tags_for_preprocessed_set()

  dict_dataset = { 
    'movielens-rating-set' : dl_movielens.data_set,
    'movielens-user-set' : dl_movielens.user_set,
    'movielens-item-set' : dl_movielens.item_set,
    'movielens-merge' : movielens_merge,
    'TMDB-preprocessed-set' : tmdb_preprocessed,
    'TMDB-movies-set' : dl_tmdb.movies_set,
    'TMDB-tags-set': tmdb_tags_set
  }
  return dict_dataset

@st.cache_resource()
def prepare_resources_for_recommendation (df: pd.DataFrame) -> Dict:
  vectors = get_vectors_from_tags(df)
  similarity = calculate_cosine_similarity_between_vectors(vectors)
  return { 'vectors':vectors, 'similarity':similarity }


dict_dataset = load_dataset(False)
df_tmdb_tags_set = dict_dataset['TMDB-tags-set']

dict_resources = prepare_resources_for_recommendation(df=df_tmdb_tags_set)


markdown = """
## Handling and Exploratory Analysis of Datasets 

This section is part of a recommendation system project. It performs various data managment tasks and visualizations.

**Data Exploration**:
- Visualizes user demographics (gender and age distribution).
- Displays top-rated movies per user.
- Shows average ratings for each movie.

**Recommendation System Basic**: Utilizes a recommendation function based on cosine similarity between movie tags.
"""

st.markdown (markdown)

st.markdown (""" 
### Movielens Dataset 

MovieLens is a widely-used benchmark dataset for collaborative filtering and recommender systems research. 
It contains ratings and metadata for movies, TV shows, and episodes
""")

df_movielens_merge = dict_dataset['movielens-merge']
df_movielens_user_set = dict_dataset['movielens-user-set']
df_movielens_item_set = dict_dataset['movielens-item-set']
df_movielens_rating_set = dict_dataset['movielens-rating-set']

col1_1,col1_2 = st.columns( [1,1] )
with col1_1:
  st.markdown ("**Counting people by gender**")
  result = df_movielens_user_set['gender'].value_counts()
  st.bar_chart (result)
with col1_2: 
  st.markdown ("**Counting people by age**")
  result = df_movielens_user_set['age'].value_counts()
  st.area_chart (result)

col2_1,col2_2 = st.columns( [1,1] )
with col2_1:
  st.markdown ("**People with the most rated movies**")
  result = df_movielens_merge.groupby( 'userID' )['rating'].count().reset_index (name='rating')
  samples = result.sort_values(by='rating',ascending=False).head(10)
  st.bar_chart ( samples['rating'] )
with col2_2:
  pass

col3_1,col3_2 = st.columns( [1,1] )

mean_rating = df_movielens_merge.groupby( 'itemID' )['rating'].mean().reset_index()
mean_rating.columns = ['itemID','mean']
mean_rating = mean_rating.sort_values(by='mean',ascending=False)
mean_rating = mean_rating.merge ( right=df_movielens_item_set, how='inner', left_on='itemID', right_on='itemID' )

with col3_1:
  st.markdown ("**Average rating for each film**")
  st.write (mean_rating[ ['itemID','name','mean'] ])  
with col3_2:
  options = mean_rating[ ['name'] ]
  selection = st.selectbox( label='Select one', options=options )
  st.write (selection)
  info = df_movielens_item_set[ df_movielens_item_set['name'] == selection ]
  genders = info.columns[ info.any() ]
  genders = [ g.replace('_', ' ') for g in genders if 'gender' in g ]
  for i,g in enumerate(genders):
    st.write(f"{i+1}. {g.capitalize()}" )



st.markdown (""" 
### TMDB-5000 Movies Dataset

The TMDB-5000 Movies dataset is a curated subset of the larger TMDB (The Movie Database) dataset, focusing on approximately 5000 high-quality movies. 
This dataset is commonly used in machine learning and natural language processing tasks related to movie analysis and recommendation systems
""")

df_tmdb_preprocessed_set = dict_dataset['TMDB-preprocessed-set']
df_tmdb_movie_set = dict_dataset['TMDB-movies-set']
df_tmdb_tags_set = dict_dataset['TMDB-tags-set']

genders = get_genders_list(df=df_tmdb_preprocessed_set)
dict_df_genders = { gender: df_tmdb_preprocessed_set[ df_tmdb_preprocessed_set['Genres'].apply( lambda x: gender in x ) ] for gender in genders }
dict_count_genders = { gender: len(df) for gender,df in dict_df_genders.items() }

col4_1, col4_2 = st.columns( [1,1] )
with col4_1:
  st.write ('**Count of movies by original languages**')
  st.bar_chart ( df_tmdb_movie_set[ 'original_language' ].value_counts() )
with col4_2:
  st.write ('**Count of movies by genre**')
  tmp = pd.DataFrame({
    'Genres' : dict_count_genders.keys(),
    'Count'  : dict_count_genders.values()
  })
  st.bar_chart(tmp, x='Genres', y='Count')

col5_1, col5_2 = st.columns( [1,1] )
df_budget_revenue = df_tmdb_movie_set[ ['title','budget','revenue'] ]
with col5_1:
  st.write ("**Display the top 10 movies with the highest revenue**")
  df_budget_revenue = df_budget_revenue.sort_values(by='revenue',ascending=False)
  st.bar_chart (df_budget_revenue.head(10), x='title', y='revenue')
with col5_2:
  st.write ("**Display the top 10 movies with the highest revenue, comparing their budgets against their revenues**")
  df_budget_revenue = df_budget_revenue.sort_values(by='revenue',ascending=False)
  st.line_chart (df_budget_revenue.head(10), x='title', y=['budget','revenue'])

