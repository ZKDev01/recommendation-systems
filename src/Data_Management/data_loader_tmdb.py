import nltk
import numpy as np  
import pandas as pd
import seaborn as sns
from typing import List, Any, Dict
from itertools import chain
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from src.data_management.utils import *

TMDB_PATH = 'dataset/tmdb/'
TMDB_CREDIT = TMDB_PATH + 'tmdb_5000_credit.csv'
TMDB_MOVIES = TMDB_PATH + 'tmdb_5000_movies.csv'

TMDB_PREPROCESSED = TMDB_PATH + 'tmdb_preprocessed.csv'

class DataLoader_TMDB:
  def __init__(self) -> None:
    self.credit_set = pd.read_csv ( TMDB_CREDIT )
    self.movies_set = pd.read_csv ( TMDB_MOVIES )

  def load_preprocessed (self):
    preprocessed_set = pd.read_csv (TMDB_PREPROCESSED)
    
    preprocessed_set['Actors'] = preprocessed_set['Actors'].apply(str_to_list)
    preprocessed_set['Genres'] = preprocessed_set['Genres'].apply(str_to_list)
    preprocessed_set['Director'] = preprocessed_set['Director'].apply(str_to_list)
    preprocessed_set['Keywords'] = preprocessed_set['Keywords'].apply(str_to_list)
  
    return preprocessed_set
  
  def preprocessing_set (self) -> None:
    merge = self.movies_set.merge ( self.credit_set, on='title' )
  
    # remove homepage, tagline columns
    merge = merge.drop ( ['homepage', 'tagline'], axis=1 )

    # fillup the missing values in the columns
    merge[ 'overview' ] = merge[ 'overview' ].fillna( '' )
    merge[ 'release_date' ] = merge[ 'release_date' ].fillna( '' )
    merge[ 'runtime' ] = merge[ 'runtime' ].fillna( '' )

    merge = merge[ [ 
      'id',
      'title',
      'overview',
      'genres',
      'keywords',
      'cast',
      'crew',
      'budget',
      'revenue'
    ]]
    
    merge[ 'genres' ] = merge[ 'genres' ].apply ( convert )
    merge[ 'keywords' ] = merge[ 'keywords' ].apply ( convert )
    merge[ 'cast' ] = merge[ 'cast' ].apply ( convert_3 )
    merge[ 'crew' ] = merge[ 'crew' ].apply ( director )

    preprocessed_set = merge.rename ( columns={ 
      'title' : 'Title',
      'overview' : 'Overview',
      'cast' : 'Actors',
      'genres' : 'Genres',
      'crew' : 'Director',
      'budget' : 'Budget',
      'revenue' : 'Revenue',
      'keywords' : 'Keywords' 
    } )

    preprocessed_set.to_csv (TMDB_PREPROCESSED, index=False)

  
  def create_tags_for_preprocessed_set (self) -> pd.DataFrame:
    preprocessed_set = self.load_preprocessed()
    preprocessed_set['Tags'] = preprocessed_set['Actors'] + preprocessed_set['Genres'] + preprocessed_set['Director'] + preprocessed_set['Keywords'] + preprocessed_set['Overview'].str.split()
    return preprocessed_set



def convert_preprocessed_set_to_list ( df: pd.DataFrame ) -> List[ str ]:
  result = df[['Title','id']]
  result['Text'] = df.apply ( lambda row: row_to_string ( row, columns=df.columns ), axis=1 )
  return result

def get_genders_list (df:pd.DataFrame) -> List[str]:
  genders:List = df['Genres'].explode().unique().tolist()
  genders.pop()
  return genders




def get_vectors_from_tags (df_tags_set: pd.DataFrame) -> Any:
  stemmer = PorterStemmer()
  
  def apply_stemmer (x):
    try:
      return [stemmer.stem(i) for i in x]
    except:
      print(x)

  df_tags_set = df_tags_set.dropna(subset=['Tags'])
  df_tags_set['Tags'] = df_tags_set['Tags'].apply(lambda x:apply_stemmer(x))
  df_tags_set['Tags'] = df_tags_set['Tags'].apply(lambda x: " ".join(x))
  
  vectorizer = CountVectorizer(max_features=5000, stop_words='english')
  vectors = vectorizer.fit_transform(df_tags_set['Tags']).toarray()

  return vectors

def calculate_cosine_similarity_between_vectors (vectors: Any) -> None:
  similarity = cosine_similarity (vectors)
  return similarity

def recommend (df: pd.DataFrame, title: str, similarity: Any, k:int = 10) -> List[str]:
  index = df[ df['Title']==title ].index[0]
  distances = similarity[index]
  movies = sorted (list(enumerate(distances)), reverse=True, key=lambda x:x[1])[:k]
  result = [ ]
  for i in movies:
    # Extract more information from df (generate a new df)
    result.append ( df.iloc[i[0]]['Title'] )
  return result





