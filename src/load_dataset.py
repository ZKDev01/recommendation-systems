import pandas as pd 
import numpy as np  


# Movielens data path
RATING_PATH = 'dataset-movielens/u.data'
ITEMS_PATH  = 'dataset-movielens/u.item'
USERS_PATH  = 'dataset-movielens/u.user'

def load_set (path: str, columns: list[str], sep: str) -> pd.DataFrame:
  df = pd.read_csv ( path, sep=sep, encoding='latin-1')
  df.columns = columns
  return df

def load_specific_set (name: str) -> pd.DataFrame: 
  if name == 'RATING':
    return load_set ( RATING_PATH, [ 'userID', 'itemID', 'rating', 'timestamp' ], '\t' )
  if name == 'USER':
    return load_set ( USERS_PATH, [ 'userID', 'age', 'gender', 'occupation', 'zipCode' ], '|' )
  if name == 'ITEMS':
    return load_set ( ITEMS_PATH, [ ], '')



def main() -> None:
  
  # Load dataset
  df_user = load_set ( USERS_PATH, [ 'userID', 'age', 'gender', 'occupation', 'zipCode' ], '|' )
  df_rating = load_set ( RATING_PATH, [ 'userID', 'itemID', 'rating', 'timestamp' ], '\t' )
  print ( df_user )
  print ( df_rating )



if __name__ == '__main__':
  main() 
