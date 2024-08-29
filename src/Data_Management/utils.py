import ast  


def convert ( obj ) -> list: 
  L = [ ]  # empty list
  for i in ast.literal_eval ( obj ):  # this function is used to convert string to list
    i[ 'name' ]
    L.append( i[ 'name' ] )    # append the name of the genre 
  return L

def convert_3 ( obj ) -> list:
  L = [ ]  
  counter = 0
  for i in ast.literal_eval ( obj ):
    if counter != 3:
      L.append ( i[ 'name' ] )
      counter += 1
    else:
      break
  return L

def director ( obj ) -> list:
  L = [ ]
  for i in ast.literal_eval ( obj ):
    if i[ 'job' ] == 'Director' :
      L.append ( i['name'] )
      break
  return L


def row_process ( row, columns: list[ str ] ) -> list[ str ]:
  
  result = [ ]
  
  for column in columns: 
    if row[ column ] == 1:
      result.append ( column )
  return result