from metrics import Metrics
from data_generator import DataGenerator


class Model: 
  def __init__(self, model, name) -> None:
    self.model = model
    self.name = name
  
  def __str__(self) -> str:
    return f'Model: { self.name }'
  
  def evaluate ( self, data: DataGenerator ): 
    trainset = data.get_trainset ( )
    testset = data.get_testset ( )

    fit_model = self.model.fit ( trainset )
    predictions = fit_model.test ( testset )

    metrics = Metrics ( predictions ).compute_metrics( 'MAE', 'RMSE' )
    
    return metrics
  

class Factory:
  def __init__(self, dataset) -> None:
    self.dataset = DataGenerator ( dataset )
    self.models: list[ Model ] = [ ]
  
  def add_model ( self, model: Model ):
    self.models.append ( model )
  
  def evaluate ( self ):
    results = { }
    for model in self.models:
      print ( f'Evaluating { model.name }' )
      results [ model.name ] = model.evaluate( self.dataset )

  def clean_models ( self ):
    self.models = [] 

