import pandas as pd 
import numpy as np  


# Movielens data path
RATING_PATH = 'dataset-movielens/u.data'
ITEMS_PATH  = 'dataset-movielens/u.item'
USERS_PATH  = 'dataset-movielens/u.user'



def load_set (name: str) -> pd.DataFrame: 
  """_summary_

  Args:
      name (str): Puede ser 'RATING', 'USER' o 'ITEM'

  Returns:
      pd.DataFrame: DataFrame de csv de la base de datos de movielens
  """

  if name == 'RATING':
    df = pd.read_csv ( RATING_PATH, sep='\t', encoding='latin-1' )
    df.columns = [ 'userID', 'itemID', 'rating', 'timestamp' ]
    # df.drop ( columns= [ 'timestamp' ] )
    return df
  
  if name == 'USER':
    df = pd.read_csv ( USERS_PATH, sep='|', encoding='latin-1' )
    df.columns = [ 'userID', 'age', 'gender', 'occupation', 'zipCode' ]
    df.drop ( columns= [ 'zipCode' ] )
    return df
  
  if name == 'ITEM':
    df = pd.read_csv ( ITEMS_PATH, sep='|', encoding='latin-1' )
    df.columns = [ 
      'itemID', 'name', 'releaseDate', 'videoReleaseDate', 'IMDbURL', 
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
      'gender_western', ]
    df.drop ( columns= [ 'videoReleaseDate' ] )
    return df

  raise Exception( 'ERROR' )




def main() -> None:
  
  # Load dataset
  df_users  = load_set ( 'USER'   )
  df_rating = load_set ( 'RATING' )
  df_items  = load_set ( 'ITEM'  )

  #print ( df_users  )
  print ( df_rating )
  #print ( df_items  )



if __name__ == '__main__':
  main() 
