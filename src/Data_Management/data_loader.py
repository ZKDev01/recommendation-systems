import numpy as np  
import pandas as pd

from src.Data_Management.utils import (
    convert,
    convert_3,
    director,
    row_process, 
    row_to_string
)

from typing import List



# Movielens data path
MOVIELENS_DATA_PATH = 'dataset/movielens_data.csv'
MOVIELENS_ITEM_PATH = 'dataset/movielens_item.csv'
MOVIELENS_USER_PATH = 'dataset/movielens_user.csv'

# TMDB data path
TMDB_CREDIT = 'dataset/tmdb_5000_credit.csv'
TMDB_MOVIES = 'dataset/tmdb_5000_movies.csv'


class DataLoader_Movielens: 
  
  def __init__(self, 
    data_path: str = MOVIELENS_DATA_PATH, 
    item_path: str = MOVIELENS_ITEM_PATH, 
    user_path: str = MOVIELENS_USER_PATH) -> None:
    
    self.DATA_PATH = data_path
    self.ITEM_PATH = item_path
    self.USER_PATH = user_path

    self.data_set = self.load_set ( 'DATA' )
    self.item_set = self.load_set ( 'ITEM' )
    self.user_set = self.load_set ( 'USER' )

  def load_set ( self, name: str ) -> pd.DataFrame:
    """
    Carga el conjunto de datos correspondiente al nombre proporcionado.

    Args:
        name (str) -> Nombre del conjunto de datos (puede ser: 'DATA', 'USER', o 'ITEM')

    Returns:
        pd.DataFrame -> El conjunto de datos cargado como DataFrame

    Raises:
        ValueError -> Si el nombre no corresponde a 'DATA', 'USER', o 'ITEM'
    """

    if name == 'DATA':
      columns = [ 'userID', 'itemID', 'rating', 'timestamp' ]
      df = pd.read_csv ( 
        self.DATA_PATH, 
        names=columns, 
        sep='\t', 
        encoding='latin-1', 
        skipinitialspace=True 
      )
      df = df.drop ( columns= [ 'timestamp' ] )
      return df
  
    if name == 'USER':
      columns = [ 'userID', 'age', 'gender', 'occupation', 'zipCode' ]
      df = pd.read_csv ( 
        self.USER_PATH, 
        names=columns, 
        sep='|', 
        encoding='latin-1', 
        skipinitialspace=True 
      )
      df = df.drop ( columns= [ 'zipCode' ] )
      return df
  
    if name == 'ITEM':
      columns = [ 
        'itemID', 
        'name', 
        'releaseDate', 
        'videoReleaseDate', 
        'IMDbURL', 
        'gender_unknown', 
        'gender_action', 
        'gender_adventure', 
        'gender_animation', 
        'gender_children', 
        'gender_comedy',
        'gender_crime',
        'gender_documentary',
        'gender_drama',
        'gender_fantasy',
        'gender_film_noir',
        'gender_horror',
        'gender_musical',
        'gender_mystery',
        'gender_romance',
        'gender_scifi',
        'gender_thriller',
        'gender_war',
        'gender_western',
      ]
      df = pd.read_csv ( 
        self.ITEM_PATH, 
        names=columns, 
        sep='|', 
        encoding='latin-1', 
        skipinitialspace=True 
      )
      df = df.drop ( columns= [ 'videoReleaseDate', 'IMDbURL' ] )
      return df

    raise ValueError ( "El nombre no corresponde a 'DATA', 'USER', o 'ITEM'" )

  def get_user_by_id ( self, id: int ) -> dict:
    """
    Obtiene información de un usuario por su ID.

    Args:
        id (int) -> ID del usuario

    Returns:
        dict -> Información del usuario como diccionario con campos 'userID', 'age', 'gender' y 'occupation'

    Raises:
        KeyError -> Si el usuario no se encuentra en la base de datos
    """
    try:
      info = self.user_set.loc [ self.user_set[ 'userID' ] == id ]
      return info[ [ 'userID', 'age', 'gender', 'occupation' ] ].iloc[0].to_dict()
    
    except KeyError:
      raise KeyError ( 'El usuario no se encuentra en la base de datos' )


  def get_item_by_id ( self, id: int ) -> dict:
    """
    Obtiene información del item por su ID.

    Args:
        id (int) -> El identificador único del item.

    Returns:
        dict -> Un diccionario conteniendo detalles del item incluyendo 'itemID', 'name',
                'releaseDate', y diversas categorías de género.

    Raises:
        KeyError -> Si no se encuentra ningún item que coincida con el ID proporcionado.
    """
    try:
      info = self.item_set.loc [ self.item_set[ 'itemID' ] == id ]
      return info[ [ 
        'itemID', 
        'name', 
        'releaseDate', 
        'gender_unknown', 
        'gender_action', 
        'gender_adventure', 
        'gender_animation', 
        'gender_children', 
        'gender_comedy',
        'gender_crime',
        'gender_documentary',
        'gender_drama',
        'gender_fantasy',
        'gender_film_noir',
        'gender_horror',
        'gender_musical',
        'gender_mystery',
        'gender_romance',
        'gender_scifi',
        'gender_thriller',
        'gender_war',
        'gender_western', ] ].iloc[0].to_dict()
    except KeyError:
      raise KeyError ( 'No se encuentra ningún item que coincida con el ID proporcionado' )
    


  def get_rating_by_ids ( self, user_id: int, item_id: int ) -> int:
    """
    Obtiene la calificación del usuario por el ID del item correspondiente.

    Args:
        user_id (int) -> El ID del usuario.
        item_id (int) -> El ID del item.

    Returns:
        int -> La calificación del usuario para el item especificado.

    Raises:
        Exception -> Si falla al recuperar la calificación.
    """
    try:
      rating = self.data_set.loc [ self.data_set[ 'userID' ] == user_id ].loc [ self.data_set[ 'itemID' ] == item_id ]
      return rating.iloc[0]['rating']
    except:
      raise Exception ( 'Fallo al recuperar la calificación' )


  def get_ratings_by_name_id ( self, column_name: str, id: int ) -> pd.DataFrame:
    """
    Filtra los datos del conjunto de ratings por el nombre de la columna y el valor del ID proporcionado.

    Args:
        column_name (str) -> El nombre de la columna para filtrar.
        id (int) -> El valor del ID para filtrar.

    Returns:
        pd.DataFrame -> Un DataFrame conteniendo solo las filas donde la columna especificada coincide con el valor del ID.

    Raises:
        Exception -> Si falla al recuperar la infomración.
    """
    try:
      filtered_data = self.data_set.loc [ self.data_set[ column_name ] == id ]
      return filtered_data [ [ 'userID', 'itemID', 'rating' ] ]
    except:
      raise Exception ( 'Fallo al recuperar la información' )

  def get_merge_by_item_ids ( self ) -> pd.DataFrame:
    """
    """

    auxiliar_set = self.item_set
    auxiliar_set = auxiliar_set.drop ( [ 'releaseDate', 'name' ], axis=1 )

    columns = [ 
      'gender_unknown', 
      'gender_action', 
      'gender_adventure', 
      'gender_animation', 
      'gender_children', 
      'gender_comedy',
      'gender_crime',
      'gender_documentary',
      'gender_drama',
      'gender_fantasy',
      'gender_film_noir',
      'gender_horror',
      'gender_musical',
      'gender_mystery',
      'gender_romance',
      'gender_scifi',
      'gender_thriller',
      'gender_war',
      'gender_western'
    ]

    auxiliar_set [ 'genders' ] = auxiliar_set.apply ( lambda row: row_process( row, columns=columns ), axis=1 )
    auxiliar_set = auxiliar_set.drop ( columns=columns, axis=1 )

    tmp_item_set = self.item_set.merge( right=auxiliar_set, how='inner', left_on='itemID', right_on='itemID' )
    tmp_item_set = tmp_item_set.drop ( columns=columns, axis=1 )

    merge = self.data_set.merge ( right=tmp_item_set, how='inner', left_on='itemID', right_on='itemID' )
    
    return merge [  merge [ 'name' ] != 'unknown' ]


class DataLoader_TMDB:
  def __init__(self, 
    credit_path: str = TMDB_CREDIT, 
    movies_path: str = TMDB_MOVIES ) -> None:
    
    self.CREDIT_PATH = credit_path
    self.MOVIES_PATH = movies_path

    self.preprocessing_set( )

  def load_set ( self ) -> None:
    self.credit = pd.read_csv ( self.CREDIT_PATH )
    self.movies = pd.read_csv ( self.MOVIES_PATH )
    
  def get_credit_dataset ( self ) -> pd.DataFrame:
    return self.credit
  
  def get_movies_dataset ( self ) -> pd.DataFrame:
    return self.movies




  def preprocessing_set ( self, verbose: bool = False ) -> None:
    # type checking: params 
    if not isinstance( verbose, bool ):
      raise TypeError( "Param 'verbose' must be boolean" )

    # 1. cargamos el dataframe para el preprocesamiento (por si antes se trabajo de forma malintencionada o si se ejecuta el codigo por segunda vez)
    self.load_set ( )

    # 2. empezar con el preprocesamiento
    
    # merge the both datasets
    merge = self.movies.merge ( self.credit, on='title' )
    
    # check the missing values ( only show if verbose is True )
    if verbose:
      print ( f'\nBefore: Missing Values: { merge.isnull().sum().sum() }' )
      print ( "... Fillup the missing values in the columns" )

    # =================================================
    
    # remove homepage, tagline columns
    merge = merge.drop ( ['homepage', 'tagline'], axis=1 )

    # fillup the missing values in the columns
    merge[ 'overview' ] = merge[ 'overview' ].fillna( '' )
    merge[ 'release_date' ] = merge[ 'release_date' ].fillna( '' )
    merge[ 'runtime' ] = merge[ 'runtime' ].fillna( '' )

    if verbose: 
      print ( f'After: Missing Values: { merge.isnull().sum().sum() }' )
    # check the missing values again

    merge = merge[ [ 
      'id',
      'title',
      'overview',
      'genres',
      'keywords',
      'cast',
      'crew',
      'budget',
      'revenue'
    ]]
    
    if verbose: 
      print ( '\n\n', merge.head( 10 ) )

    # Processing columns
    ## GENRES

    if verbose:
      print ( '\n\nProcessing columns: GENRES' )
      print ( merge[ 'genres' ].head( 10 ) )

    merge[ 'genres' ] = merge[ 'genres' ].apply ( convert )

    if verbose:
      print ( '\n\nResults: GENRES' )
      print ( merge[ ['title', 'genres'] ].head( 10 ) )

    ## KEYWORDS

    if verbose:
      print ( '\n\nProcessing columns: KEYWORDS' )
      print ( merge[ 'keywords' ].head( 10 ) )

    merge[ 'keywords' ] = merge[ 'keywords' ].apply ( convert )

    if verbose:
      print ( '\n\nResults: KEYWORDS' )
      print ( merge[['title', 'keywords']].head( 10 ) )

    ## CAST

    if verbose:
      print ( '\n\nProcessing columns: CAST' )
      print ( merge[ 'cast' ].head( 10 ) )
      
    merge[ 'cast' ] = merge[ 'cast' ].apply ( convert_3 )

    if verbose:
      print ( '\n\nResults: CAST' )
      print ( merge[['title', 'cast']].head( 10 ) )

    ## CREW

    if verbose:
      print ( '\n\nProcessing columns: CREW' )
      print ( merge[ 'crew' ].head( 10 ) )
      
    merge[ 'crew' ] = merge[ 'crew' ].apply ( director )

    if verbose:
      print ( '\n\nResults: CREW' )
      print ( merge[['title', 'crew']].head( 10 ) )

    self.preprocessed_set = merge.rename ( columns={ 
      'title' : 'titulo',
      'overview' : 'resumen',
      'cast' : 'actores',
      'genres' : 'genders',
      'crew' : 'director',
      'budget' : 'presupuesto',
      'revenue' : 'ingresos' 
    } )




  def get_preprocessed_set ( self ) -> pd.DataFrame:
    return self.preprocessed_set



  def convert_preprocessed_set_to_list ( self ) -> List[ str ]:
    list_movies: List[ str ] = self.get_preprocessed_set ( ).values.tolist( )
    auxiliar_set = self.get_preprocessed_set ( )
    auxiliar_set [ 'text' ] = auxiliar_set.apply ( lambda row: row_to_string ( row, columns=auxiliar_set.columns ), axis=1 )
    
    return auxiliar_set [ 'text' ].values.tolist ( )
  

