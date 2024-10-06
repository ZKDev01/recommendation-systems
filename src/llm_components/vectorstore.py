import os

from src.llm_components.utils import get_embedding

from typing import List,Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document


FAISS_PATH = os.getcwd() + '\\database\\faiss'


class Faiss_Vectorstore:

  def __init__(self, movies:Dict[str,str]) -> None:
    embedding = get_embedding()

    movies_like_documents: List[ Document ] = [ 
      Document(page_content=description,metadata={'Title':title}) 
      for title,description in movies.items() 
    ]

    self.__vectorstore = FAISS.from_documents ( 
      documents=movies_like_documents,
      embedding=embedding
    )

  def similarity_search ( self, query:str, k:int=5 ) -> List[Document]:
    results = self.__vectorstore.similarity_search ( query=query, k=k )
    return results
