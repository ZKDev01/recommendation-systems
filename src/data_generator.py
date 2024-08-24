from surprise.model_selection import (
  train_test_split
)


""" TODO

def train_test_split () -> Any:
  pass

"""


class DataGenerator: 
  def __init__(self, data, percentage = 0.25) -> None:
    """Data Generator

    Build a 75/25 train/test split for measuring accuracy

    Args:
        percentage (float, optional): _description_. Defaults to 0.25.
    """
    self.trainset, self.testset = train_test_split ( data, test_size=percentage, random_state=1 )

# MEJORAR CON UN METODO PROPIO PARA EL TRAIN-TEST SPLIT con los DATAFRAME y RANDOM 

  def get_trainset( self ):
    return self.trainset
  
  def get_testset( self ):
    return self.testset