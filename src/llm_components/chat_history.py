from src.llm_components.utils import *
from src.llm_components.vectorstore import *


from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder





class ChatHistory:
  def __init__(self) -> None:
    self.model: GoogleGenerativeAI = get_model ( )
    self.embedding: GoogleGenerativeAIEmbeddings = get_embedding ( )
    self.chat: list = [ ]

    self.prompt = """
    Eres un asistente, capaz de recomendar peliculas dado una consulta del usuario. 
    Tambien debes tener conocimientos sobre la conversacion que tengas con el usuario. 
    Eres capaz ademas de recomendar, dar una explicacion de las peliculas segun 
    la informacion que se te proporcione de la pelicula  
    """


  def make_chain ( self ) -> None:
    self.prompt = ChatPromptTemplate.from_messages ( [
      ( 'system', f'{self.prompt}' ),
      MessagesPlaceholder ( variable_name='chat' ),
      ( 'human', '{input}' )
    ] )
    self.chain = self.prompt | self.model

  def clean_chat ( self ) -> None:
    self.chat : list = [ ]
  

  def to_answer_query ( self, query: str, movies: list[ str ] ) -> str:
    # movies = movies[ 0:20 ]

    self.chat.append( 
      HumanMessage ( content='A continuacion vas a recibir un conjunto de peliculas para que puedas recomendar en respuesta a la pregunta del usuario' )
    ) 

    for movie in movies:
      self.chat.append ( HumanMessage ( content=movie ) )

    self.chat.append(
      HumanMessage ( content='Este conjunto de peliculas son las opciones que puedes considerar para recomendar' )
    )

    preprocessed_query = f"""
    Pregunta del usuario: { query }
    """
    response = self.chain.invoke ( {
      'input'  : preprocessed_query,
      'chat'   : self.chat
    } )

    self.chat.append ( HumanMessage( content=query ) )
    self.chat.append ( AIMessage( content=response ) )

    return response
