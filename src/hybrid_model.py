
from surprise import AlgoBase
from model_factory import Model




class HybridModel ( AlgoBase ):

  def __init__ (self, models, weights, **kwargs):
    super().__init__(**kwargs) 
    self.models: list[ Model ] = models
    self.weights = weights
  
  def fit (self, trainset):
    AlgoBase.fit ( self, trainset )
    for model in self.models:
      model.model.fit ( trainset )
    return self
  
  def estimate ( self, user_id, item_id ):
    scores = 0 
    weight = 0
    for i in range ( len( self.models ) ):
      scores += self.models[i].model.predict ( user_id, item_id ).est * self.weights[i]
      weight += self.weights[i]

    return scores/weight