from collections import defaultdict

import streamlit as st
import pandas as pd 
import numpy as np  

from surprise import ( 
  Prediction,
  KNNBaseline,
  SVD,
  KNNWithMeans
)

import matplotlib.pyplot as plt

from src.data_management.utils import *
from src.data_management.data_loader import *
from src.data_management.exploratory_data_analysis import *

from src.llm_components.utils import *
from src.llm_components.vectorstore import *
from src.llm_components.chat_history import *

from src.recsys_analysis.utils import *
from src.recsys_analysis.metrics import *
from src.recsys_analysis.model_factory import *
from src.recsys_analysis.data_generator import *




if __name__ == '__main__':
  st.title ('New Process')




