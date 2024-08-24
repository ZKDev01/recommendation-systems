import pandas as pd

from surprise import Dataset, Reader
from surprise.model_selection import train_test_split


class DataGenerator: 
  def __init__(self, dataframe: pd.DataFrame, percentage: float = 0.25, rating_scale = (1, 5) ) -> None:
    
    self.dataframe = dataframe
    self.percentage = percentage
    self.rating_scale = rating_scale

  def train_test_split ( self ):    
    self.trainset, self.testset = train_test_split ( 
      data=self.dataset,
      test_size=self.percentage,
      random_state=1
    )  

  def from_df_to_dataset ( self, columns: list[str] = [ 'userID', 'itemID', 'rating' ] ) -> Dataset:
    self.dataset = Dataset.load_from_df ( self.dataframe [ columns ], Reader ( rating_scale=self.rating_scale ) )

  def get_trainset( self ):
    return self.trainset
    
  def get_testset( self ):
    return self.testset



def test ( ) -> None:  
  print ( '======= Testing =======' )
  
  pass

if __name__ == '__main__':
  test ( )

