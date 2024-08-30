import os

from src.LLM.utils import get_model, get_embedding 

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document


FAISS_PATH = os.getcwd() + '\\database\\faiss'


class Faiss_Vectorstore:

  def __init__(self, movies: List[str], load: bool = False) -> None:
    """
    Inicializa una instancia de `Faiss_Vectorstore` para manejar y buscar en una base de datos vectorial de FAISS

    Este constructor permite opcionalmente cargar la base de datos vectorial existente si se pasa `True` al parametro `load`. 
    Si `load` es `False` (valor predeterminado), se crea una nueva base de datos vectorial con la listra proporcionada

    Args:
        load (bool, optional): Indica si se debe cargar la base de datos vectorial. Defaults to False.
    """
    embedding = get_embedding()

    if load:
      self.__vectorstore = FAISS.load_local( 
        folder_path=FAISS_PATH,
        embeddings=embedding,
        allow_dangerous_deserialization=True
      )
    else:
      
      movies_like_documents: List[ Document ] = [ Document(movie) for movie in movies ] 

      # Problema con la carga de peliculas 
      movies_like_documents = movies_like_documents[0:499]

      self.__vectorstore = FAISS.from_documents (
        documents=movies_like_documents[0 : 99],
        embedding=embedding
      )


      for i in range ( 1, 5 ):
        extension = FAISS.from_documents ( 
          documents=movies_like_documents[i*100 : i*100+99],
          embedding=embedding
        )
        self.__vectorstore.merge_from ( extension )
      

      """ 
      # SE DEMORA MUCHO ESTO
      counter = 1
      for item in movies_like_documents:
        
        extension = FAISS.from_documents (
          documents=[ item ],
          embedding=embedding
        )
        self.__vectorstore.merge_from ( extension )
        print ( counter )
        counter += 1
      """


  def similarity_search ( self, query: str, k: int = 10 ) -> list[str]:
    """
    Realiza una busqueda de similaridad en la base de datos vectorial utilizando una cadena de texto 

    Args:
        query (str): la consulta de texto para realizar la busqueda de similitud
        k (int, optional): numero de resultados similares a retornar. Defaults to 3.

    Returns:
        list[str]: una lista de cadenas donde cada elemento es el contenido de un documento encontrado que es similar a la consulta
    """
    if not k > 0:
      raise Exception( 'El parametro k no puede ser negativo ni 0' )
    results = self.__vectorstore.similarity_search ( query=query, k=k )
    results = [ r.page_content for r in results ]
    return results

