import pandas as pd
import numpy as np  

import surprise
import time
import sklearn

from sklearn.metrics.pairwise import sigmoid_kernel
from matplotlib import pyplot as plt  
from surprise.model_selection import cross_validate

from pandas import DataFrame

from data_loader import load_set





# USER BASED COLLABORATIVE FILTERING

class UserBased_CollaborativeFiltering (): 
  def __init__(self, users: DataFrame, items: DataFrame, ratings: DataFrame, topk = 10, max_rating = 5.0) -> None:
    self.users: DataFrame = users
    self.items: DataFrame = items
    
    self.ratings: DataFrame = ratings

    self.topk = topk
    self.max_rating = max_rating

  def normalize ( self, dataframe: DataFrame ):
    # sum entries of rows
    row_sum_ratings = dataframe.sum ( axis=1 )
    print ( f'Suma de los ratings {row_sum_ratings}' )

    # count non-zero entries of rows
    non_zero_count = dataframe.astype ( bool ).sum ( axis=1 )

    # mean of rows
    dataframe_mean = row_sum_ratings / non_zero_count 

    # return a dataframe: substract on rows
    return dataframe.subtract ( dataframe_mean, axis=0 ) 



  def __str__(self) -> str:
    string = f""" 
    DATAFRAMES
    =======================================
      USERS:
      { self.users }
    =======================================
      ITEMS: 
      { self.items }
    =======================================
      RATING: 
      { self.ratings }
    """
    return string




class ItemBased_CollaborativeFiltering ():
  pass






def test () -> None:
  # USER BASED COLLABORATIVE FILTERING
  users = load_set( 'USER' )
  items = load_set( 'ITEM' )
  ratings = load_set( 'RATING' )

  userbased_cf = UserBased_CollaborativeFiltering( users=users, items=items, ratings=ratings )
  # print( userbased_cf )

  print ( "Normalizacion de los ratings" )
  print ( userbased_cf.normalize ( userbased_cf.ratings ) )

if __name__ == '__main__':
  test()
