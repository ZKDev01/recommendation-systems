from typing import Any
from collections import defaultdict

from surprise import ( 
  Prediction,
  KNNBaseline,
  SVD,
  KNNWithMeans,
)




def get_top_n ( predictions: list[Prediction], user_id: int, n: int = 10 ) -> defaultdict [ Any, list ]:
  """_summary_

  Args:
      predictions (list[Prediction]): _description_
      user_id (int): _description_
      n (int, optional): _description_. Defaults to 10.

  Returns:
      defaultdict [ Any, list ]: _description_
  """
  
  top_n = defaultdict ( list )

  for element in predictions:
    if user_id == element.uid:
      top_n [ user_id ].append ( ( element.iid, element.est ) )

  for user_id, user_ratings in top_n.items ( ):
    user_ratings.sort ( key=lambda x: x[1], reverse=True )
    top_n [ user_id ] = user_ratings [ :n ]

  return top_n


