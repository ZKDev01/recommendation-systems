from surprise import ( 
  accuracy
) 


class Metrics: 
  def __init__(self, predictions) -> None:
    self.predictions = predictions
    self.metrics = { }

  def compute_metrics ( self, *args ) -> dict :
    if 'MAE' in args: self.MAE ( )
    if 'RMSE' in args: self.RMSE ( )
    return self.metrics

  def MAE  ( self ):
    self.metrics [ 'MAE' ] = accuracy.mae ( self.predictions ) 

  def RMSE ( self ): 
    self.metrics [ 'RMSE' ] = accuracy.rmse ( self.predictions ) 