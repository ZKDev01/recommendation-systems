import streamlit as st

st.set_page_config(
  page_title='Home Page',
  layout='wide'
)

def main () -> None:
  st.title ('Sistemas de Recomendacion')
  with st.expander ('**About**'):
    st.write ('Proyecto de Sistema de Recuperacion de Informacion')

if __name__ == '__main__':
  main ()
