import numpy as np  
import pandas as pd


# Movielens data path
MOVIELENS_DATA_PATH = 'dataset/movielens_data.csv'
MOVIELENS_ITEM_PATH = 'dataset/movielens_item.csv'
MOVIELENS_USER_PATH = 'dataset/movielens_user.csv'


class DataLoader_Movielens: 
  
  def __init__(self, 
    data_path: str = MOVIELENS_DATA_PATH, 
    item_path: str = MOVIELENS_ITEM_PATH, 
    user_path: str = MOVIELENS_USER_PATH) -> None:
    
    self.DATA_PATH = data_path
    self.ITEM_PATH = item_path
    self.USER_PATH = user_path

    self.data_set = self.load_set ( 'DATA' )
    self.item_set = self.load_set ( 'ITEM' )
    self.user_set = self.load_set ( 'USER' )

  def load_set (self, name: str ) -> pd.DataFrame:
    """Esta funcion se encarga de dado una cadena de texto, ya sea: 'DATA', 'USER', 'ITEM' poder retornar el dataset correspondiente 

    Args:
        name (str): nombre del dataset y puede ser 'DATA', 'USER', 'ITEM'

    Returns:
        pd.DataFrame: _description_
    """

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



  def get_user_by_id ( self, id: int ):
    """_summary_

    Args:
        id (int): _description_

    Returns:
        _type_: _description_
    """

    info = self.user_set.loc [ self.user_set[ 'userID' ] == id ]
    return info[ [ 'userID', 'age', 'gender', 'occupation' ] ].iloc[0].to_dict()


  def get_item_by_id ( self, id: int ):
    """_summary_

    Args:
        id (int): _description_

    Returns:
        _type_: _description_
    """
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
    """_summary_

    Args:
        user_id (int): _description_
        item_id (int): _description_

    Returns:
        _type_: _description_
    """
    try:
      rating = self.data_set.loc [ self.data_set[ 'userID' ] == user_id ].loc [ self.data_set[ 'itemID' ] == item_id ]
      return ( rating.iloc[0]['rating'], True )
    except:
      # Failed to retrieve the rating
      return ( -1, False )


  def get_ratings_by_name_id ( self, column_name: str, id: int ):
    """_summary_

    Args:
        column_name (str): _description_
        id (int): _description_

    Returns:
        _type_: _description_
    """
    filtered_data = self.data_set.loc [ self.data_set[ column_name ] == id ]
    return filtered_data [ [ 'userID', 'itemID', 'rating' ] ]
  





class DataLoader_TMDB:
  pass

