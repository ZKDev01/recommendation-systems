from src.llm_components.utils import *
from src.llm_components.vectorstore import *


from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder



base_prompt = """ 
You are an AI assistant specializing in movie and series analysis. Your task is to respond to user queries about films or television shows while incorporating relevant information from provided documents using Retrieval-Augmented Generation (RAG).

When responding to a user's request:

- Directly address the specific query or topic presented in the user's input.
- Provide an analysis or discussion related to movies or TV series.
- Utilize any provided information within your response, effectively integrating them through RAG functionality.

Your responses should be informative, engaging, and tailored to the user's interests in cinema and television. When referencing external information, seamlessly incorporate the key points or insights they contain into your analysis.

For example, if a user asks about the plot of "Inception" and provides a document discussing its themes, your response might analyze both the film itself and the themes discussed in the document, connecting them where appropriate.

Remember to maintain a conversational tone while providing accurate and insightful content about movies and series, always keeping the user's query at the forefront of your analysis.
"""



class ChatHistory:
  def __init__(self) -> None:
    self.model: GoogleGenerativeAI = get_model ( )
    self.embedding: GoogleGenerativeAIEmbeddings = get_embedding ( )
    self.chat: list = [ ]

    self.prompt = ChatPromptTemplate.from_messages(
      [ 
        ('system',f'{base_prompt}'),
        MessagesPlaceholder (variable_name='chat'),
        ('human', '{input}')
      ]
    )
    self.chain = self.prompt|self.model

  def clean_chat ( self ) -> None:
    self.chat : list = [ ]


  def to_process_query ( self, query:str, movies:List[str], record:List[str] ) -> str:
    documents = '\n'.join(movies)
    input_query = f"""
    query: {query} 

    information: 
    {documents}

    record: 
    {record}
    """

    response = self.chain.invoke ({
      'input' : input_query,
      'chat'  : self.chat 
    })
    
    self.chat.append ( HumanMessage( content=input_query ) )
    self.chat.append ( AIMessage( content=response ) )
    return response
