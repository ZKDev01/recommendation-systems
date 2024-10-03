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


st.markdown ( 
'''
## Modelos de Recomendación

A continuación se presentan comparaciones y análisis de los modelos tradicionales de sistemas de recomendación usando como conjunto de datos para en entrenamiento y la evaluación Movielens  

'''
)

dl_movielens = DataLoader_Movielens ( )
data_set = dl_movielens.data_set

data_generator = DataGenerator ( 
  dataframe=data_set
)
data_generator.from_df_to_dataset()
data_generator.train_test_split()

st.markdown (  
'''
### Comparación y Evaluación de los Diferentes Modelos provistos por Surprise
''')  
# results = get_evaluation_and_comparison_of_machine_learning_models( data_set )

# st.write ( results )
  
st.markdown (  
'''
Analizando los modelos KNN Baseline y SVD 'clasico'
''')

model_knn_baseline = Model ( 
  model=KNNBaseline(),
  name='KNN Baseline'
)
metrics = model_knn_baseline.evaluate ( data=data_generator )
metrics.compute_metrics ( 'MAE', 'RMSE' )

st.write ( '===============================' )
st.write ( model_knn_baseline )
st.write ( metrics )
  
model_svd = Model (
  model=SVD(),
  name='SVD'
)
metrics = model_svd.evaluate ( data=data_generator )
metrics.compute_metrics ( 'MAE', 'RMSE' )

st.write ( '===============================' )
st.write ( model_svd )
st.write ( metrics )

st.markdown (  
'''
### Sistema de Recomendación Híbrido  
'''
)

hybrid_model = HybridModel_Weighted (
  name='SVD x KNN with Means',
  models=[
    Model(
      model=SVD(),
      name='SVD'
    ),
    Model(
      model=KNNWithMeans ( sim_options= { 'name': 'cosine', 'user_based': False } ),
      name='KNN with Means'
    )
  ],
  weights=[0.5,0.5]
)

st.write ( hybrid_model )
  
_, testset = data_generator.get_train_test_set ( )
hybrid_model.fit ( data_generator )
predictions = hybrid_model.test ( testset )

metrics = Metrics ( predictions=predictions )
metrics.compute_metrics ( 'RMSE', 'MAE' )

st.write ( metrics )

# PEDIR AL USUARIO QUE INSERTE UN USER_ID 
top_n = get_top_n ( predictions=predictions, user_id=10, n=10 )
st.write ( top_n )
