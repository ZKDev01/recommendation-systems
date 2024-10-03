from typing import Any
from collections import defaultdict

import pandas as pd 

from surprise import ( 
  Reader,
  Dataset,
  NormalPredictor,
  KNNBasic,
  KNNWithMeans,
  KNNBaseline,
  KNNWithZScore,
  SVD,
  BaselineOnly,
  SVDpp,
  NMF,
  SlopeOne,
  CoClustering,
  accuracy, 
  Prediction
)

from matplotlib import pyplot as plt

from surprise.model_selection import cross_validate, train_test_split
from surprise.prediction_algorithms import AlgoBase
from sklearn.metrics.pairwise import sigmoid_kernel


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


def get_evaluation_and_comparison_of_machine_learning_models ( df: pd.DataFrame, rating_scale=(1,5) ) -> pd.DataFrame:
  """_summary_

  Args:
      df (pd.DataFrame): _description_
      rating_scale (tuple, optional): _description_. Defaults to (1,5).

  Returns:
      pd.DataFrame: _description_
  """
  
  reader = Reader ( rating_scale=rating_scale )
  data = Dataset.load_from_df ( df, reader )
  
  benchmark = { }
  algorithms = { 
  'SVD' : SVD(),
  'SVD++' : SVDpp(),
  'Slope One': SlopeOne(),
  'NMF': NMF(),
  'Normal Predictor': NormalPredictor(),
  'KNN Baseline': KNNBaseline(),
  'KNN with Means': KNNWithMeans(),
  'KNN Basic': KNNBasic(),
  'KNN with ZScore': KNNWithZScore(),
  'Baseline Only': BaselineOnly(),
  'CoClustering': CoClustering()
  }
  
  marks = { 
    'test_rmse' : 'Test RMSE',
    'fit_time'  : 'Fit Time',
    'test_time' : 'Test Time'
  }

  for algorithm in algorithms.keys():
    results = cross_validate ( algorithms[algorithm], data, measures= ['RMSE'], cv=5, verbose=False )
    tmp = { }
    for key in marks.keys():
      tmp [ marks[key] ] = pd.Series ( results [ key ] ).mean()

    benchmark [ algorithm ] = tmp

  surprise_results = pd.DataFrame.from_dict ( benchmark ).T
  return surprise_results