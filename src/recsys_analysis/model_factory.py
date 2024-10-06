from typing import List,Dict
from surprise import AlgoBase,Prediction

from src.recsys_analysis.utils import *



class DataGenerator: 
  def __init__(self, dataframe: pd.DataFrame, percentage: float = 0.25, rating_scale = (1, 5) ) -> None:
    self.dataframe = dataframe
    self.percentage = percentage
    self.rating_scale = rating_scale
    self.from_df_to_dataset()
    self.train_test_split()
  
  def train_test_split ( self ) -> None:        
    self.trainset, self.testset = train_test_split ( 
      data=self.dataset,
      test_size=self.percentage,
      random_state=1
    )  

  def from_df_to_dataset ( self, columns: list[str] = [ 'userID', 'itemID', 'rating' ] ) -> Dataset:
    self.dataset = Dataset.load_from_df ( self.dataframe [ columns ], Reader ( rating_scale=self.rating_scale ) )

  def get_train_test_set ( self ) -> Any:
    return self.trainset, self.testset 



class Metrics:   
  def __init__(self, predictions: List[Prediction]) -> None:
    self.predictions:List[Prediction] = predictions
    
  def compute_metrics (self) -> dict :
    metrics:Dict = { }
    metrics['MAE'] = accuracy.mae (self.predictions)
    metrics['RMSE'] = accuracy.rmse (self.predictions)
    return metrics



class Model: 
  def __init__(self, model: AlgoBase, name: str) -> None:
    self.model = model
    self.name = name
    self.is_fit = False
  
  def __repr__(self) -> str:
    return self.name

  def fit ( self, data: DataGenerator ) -> AlgoBase:
    trainset, _ = data.get_train_test_set ( )
    fit_model = self.model.fit ( trainset )
    self.is_fit = True
    return fit_model

  def evaluate ( self, data: DataGenerator ) -> Metrics: 
    fit_model = self.fit ( data )
    _, testset = data.get_train_test_set ( )
    predictions = fit_model.test ( testset, verbose=False )

    return Metrics ( predictions )

  def predict ( self, userID: int, itemID: int, r_ui: float = None, verbose: bool = False ) -> Prediction:
    return self.model.predict ( uid=userID, iid=itemID, r_ui=r_ui, verbose=False )



class HybridModel_Weighted ( AlgoBase ):

  def __init__ (self, name: str, models: list[ Model ], weights: list[ float ], **kwargs):
    super().__init__(**kwargs) 
    self.name = name
    self.models = models
    self.weights = weights
  
  def __repr__(self) -> str:
    return self.name

  def get_trainset (self) -> Any:
    return self.trainset

  def fit (self, data_generator: DataGenerator) -> 'HybridModel_Weighted':
    self.trainset, _ = data_generator.get_train_test_set( )
    AlgoBase.fit ( self, self.trainset )
    for model in self.models:
      model.fit ( data_generator )
    return self
  
  def estimate ( self, user_id: int, item_id: int ) -> float:
    scores = 0
    for k in range ( len( self.models ) ):
      model_k = self.models[ k ]
      weight_k = self.weights[ k ]
      predition = model_k.predict ( userID=user_id, itemID=item_id, verbose=False )
      estimated_rating = predition.est
      scores += weight_k * estimated_rating
    return scores



