from surprise import ( 
  accuracy
) 


class Metrics: 
  """_summary_
  """
  
  def __init__(self, predictions) -> None:
    """_summary_

    Args:
        predictions (_type_): _description_
    """
    self.predictions = predictions
    self.metrics = { }

  def compute_metrics ( self, *args ) -> dict :
    """_summary_

    Returns:
        dict: _description_
    """
    if 'MAE' in args: self.MAE ( )
    if 'RMSE' in args: self.RMSE ( )
    return self.metrics

  def MAE  ( self ):
    """_summary_
    """
    self.metrics [ 'MAE' ] = accuracy.mae ( self.predictions ) 

  def RMSE ( self ): 
    """_summary_
    """
    self.metrics [ 'RMSE' ] = accuracy.rmse ( self.predictions ) 

  def __str__(self) -> str:
    return f'MAE: { self.metrics['MAE'] }\nRMSE: { self.metrics['RMSE'] }'