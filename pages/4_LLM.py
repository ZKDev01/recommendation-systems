import numpy as np  
import pandas as pd 
import streamlit as st

from src.data_management.utils import *
from src.data_management.data_loader_tmdb import *
from src.data_management.data_loader_movielens import *

from src.llm_components.utils import *
from src.llm_components.vectorstore import *
from src.llm_components.chat_history import *

from src.recsys_analysis.utils import *
from src.recsys_analysis.model_factory import *


st.set_page_config(
  page_title='RecSys using LLM',
  layout='wide'
)


# region: CACHE

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
def compute_similarity (df:pd.DataFrame):
  vectors = get_vectors_from_tags(df)
  similarity = calculate_cosine_similarity_between_vectors(vectors)
  return similarity

@st.cache_resource()
def convert_columns_to_str (df:pd.DataFrame) -> pd.DataFrame:
  df_tmdb_str = convert_preprocessed_set_to_list(df=df)
  return df_tmdb_str

# endregion 

def get_str_from_title_name (df:pd.DataFrame,title:str) -> str:
  result = df[ df['Title'] == title ]
  output:str = f"""
  **Title**: {result['Title'].iloc[0]} \n
  **Overview**: {result['Overview'].iloc[0]} \n
  **Director**: { ', '.join(result['Director'].iloc[0]) } \n
  **Genres**: { ', '.join(result['Genres'].iloc[0]) } \n
  **Actors**: { ', '.join(result['Actors'].iloc[0]) } \n
  **Budget**: {result['Budget'].iloc[0]} \n
  **Revenue**: {result['Revenue'].iloc[0]}
  """ 
  return output



st.title ('Recommendation System using LLM and RAG')

with st.expander(label='**About**'):
  st.markdown(
    """
    The TMDB 5000 Movies is used on this page

    In this page include:

    - *LLM Seq Sim Implementation*: Generates semantic item embeddings using Large Language Models (LLMs)
    - *RAG*: Retrieves relevant information from a knowledge base to augment LLM embeddings
    """
  )



# load datasets
dict_dataset = load_dataset()

df_tmdb_preprocessed = dict_dataset['TMDB-preprocessed-set']
df_tmdb_tags_set = dict_dataset['TMDB-tags-set']

similarity = compute_similarity(df_tmdb_tags_set)

chat_history = ChatHistory()



# Inputs
col1_1,col1_2 = st.columns([4,3])
with col1_1:
  selection = st.multiselect(label='Historial',options=df_tmdb_tags_set['Title'])
with col1_2:
  query = st.text_input(label='Query')



# Buttons
btn_recommend = st.button ('Recommend')



# Show answer y process the inputs
if btn_recommend and len(selection) > 0 and len(query) > 0:
  recommendation:set = set()
  recommendation.update (selection)
  for s in selection:
    recommendation.update (recommend (df=df_tmdb_tags_set, title=s, similarity=similarity, k=20))
  
  #st.write (query)
  #st.write (len(recommendation))
  
  col2_1,col2_2,col2_3 = st.columns([1,1,1])
  with col2_1:
    st.write (f"Total: {len(recommendation)}")

    dict_title_tags = { title:', '.join(df_tmdb_tags_set[ df_tmdb_tags_set['Title'] == title ]['Tags'].iloc[0]) for title in recommendation }
    for title in dict_title_tags.keys():
      with st.expander (label=f"**Movie**: {title}"):
        st.markdown (get_str_from_title_name(df_tmdb_preprocessed,title))

  with col2_2:
    
    faiss = Faiss_Vectorstore(movies=dict(list(dict_title_tags.items())[:50]))
    result:List[Document] = faiss.similarity_search(query=query,k=10)
    
    st.write ('**Movies similar to the query**')
    for document in result:
      title:str = document.metadata['Title']
      with st.expander (label=f"**Movie**: {title}"):
        st.markdown (get_str_from_title_name(df_tmdb_preprocessed,title))

  with col2_3:
    st.write ('**To answer query using LLM**')
    
    movies = [ get_str_from_title_name(df_tmdb_preprocessed,document.metadata['Title']) for document in result ]
    answer = chat_history.to_process_query(query=query,movies=movies,record=selection)
    st.write (answer)
  
