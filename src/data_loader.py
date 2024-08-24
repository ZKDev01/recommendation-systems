import os
import numpy as np  
import pandas as pd

from surprise import ( 
  Dataset,
  Reader,
  accuracy, 
  SVD,
  AlgoBase,
  BaselineOnly,
  NormalPredictor,
  KNNBasic,
  KNNWithMeans,
  KNNBaseline,
  KNNWithZScore,
  SVDpp,
  NMF,
  SlopeOne,
  CoClustering,
  accuracy
) 

from surprise.model_selection import (
  train_test_split
)


# Movielens data path
DATA_PATH = 'dataset/data.csv'
ITEM_PATH = 'dataset/item.csv'
USER_PATH = 'dataset/user.csv'


class DataLoader: 
  
  def __init__(self, 
    data_path: str = DATA_PATH, 
    item_path: str = ITEM_PATH, 
    user_path: str = USER_PATH) -> None:
    
    self.DATA_PATH = data_path
    self.ITEM_PATH = item_path
    self.USER_PATH = user_path

    self.data_set = self.load_set ( 'DATA' )
    self.item_set = self.load_set ( 'ITEM' )
    self.user_set = self.load_set ( 'USER' )

  def load_set (self, name: str ) -> pd.DataFrame:

    if name == 'DATA':
      columns = [ 'userID', 'itemID', 'rating', 'timestamp' ]
      df = pd.read_csv ( 
        self.DATA_PATH, 
        names=columns, 
        sep='\t', 
        encoding='latin-1', 
        skipinitialspace=True 
      )
      df = df.drop ( columns= [ 'timestamp' ] )
      return df
  
    if name == 'USER':
      columns = [ 'userID', 'age', 'gender', 'occupation', 'zipCode' ]
      df = pd.read_csv ( 
        self.USER_PATH, 
        names=columns, 
        sep='|', 
        encoding='latin-1', 
        skipinitialspace=True 
      )
      df = df.drop ( columns= [ 'zipCode' ] )
      return df
  
    if name == 'ITEM':
      columns = [ 
        'itemID', 
        'name', 
        'releaseDate', 
        'videoReleaseDate', 
        'IMDbURL', 
        'gender_unknown', 
        'gender_action', 
        'gender_adventure', 
        'gender_animation', 
        'gender_children', 
        'gender_comedy',
        'gender_crime',
        'gender_documentary',
        'gender_drama',
        'gender_fantasy',
        'gender_film_noir',
        'gender_horror',
        'gender_musical',
        'gender_mystery',
        'gender_romance',
        'gender_scifi',
        'gender_thriller',
        'gender_war',
        'gender_western',
      ]
      df = pd.read_csv ( 
        self.ITEM_PATH, 
        names=columns, 
        sep='|', 
        encoding='latin-1', 
        skipinitialspace=True 
      )
      df = df.drop ( columns= [ 'videoReleaseDate', 'IMDbURL' ] )
      return df

  def load_dataset ( self ) -> Dataset:
    reader = Reader ( rating_scale= ( 1,5 ) )
    data = Dataset.load_from_df ( self.data_set [ [ 'userID', 'itemID', 'rating' ] ], reader )
    return data

  def get_user_by_id ( self, id: int ):
    info = self.user_set.loc [ self.user_set[ 'userID' ] == id ]
    return info[ [ 'userID', 'age', 'gender', 'occupation' ] ].iloc[0].to_dict()

  def get_item_by_id ( self, id: int ):
    info = self.item_set.loc [ self.item_set[ 'itemID' ] == id ]
    return info[ [ 
      'itemID', 
      'name', 
      'releaseDate', 
      'gender_unknown', 
      'gender_action', 
      'gender_adventure', 
      'gender_animation', 
      'gender_children', 
      'gender_comedy',
      'gender_crime',
      'gender_documentary',
      'gender_drama',
      'gender_fantasy',
      'gender_film_noir',
      'gender_horror',
      'gender_musical',
      'gender_mystery',
      'gender_romance',
      'gender_scifi',
      'gender_thriller',
      'gender_war',
      'gender_western', ] ].iloc[0].to_dict()

  def get_rating_by_ids ( self, user_id: int, item_id: int ):
    try:
      rating = self.data_set.loc [ self.data_set[ 'userID' ] == user_id ].loc [ self.data_set[ 'itemID' ] == item_id ]
      return ( rating.iloc[0]['rating'], True )
    except:
      # Failed to retrieve the rating
      return ( -1, False )

  def get_ratings_by_name_id ( self, column_name: str, id: int ):
    filtered_data = self.data_set.loc [ self.data_set[ column_name ] == id ]
    return filtered_data [ [ 'userID', 'itemID', 'rating' ] ]
  



def test() -> None:
  pass


if __name__ == '__main__':
  test() 
