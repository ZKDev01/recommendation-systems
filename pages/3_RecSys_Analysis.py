import numpy as np  
import pandas as pd 
import streamlit as st
from typing import Set,List,Dict

from src.data_management.utils import *
from src.data_management.data_loader_tmdb import *
from src.data_management.data_loader_movielens import *

from src.recsys_analysis.utils import *
from src.recsys_analysis.model_factory import *

# region: CACHE

@st.cache_resource()
def load_dataset ( generate_preprocessing:bool=False ) -> Dict[str,pd.DataFrame]:
  dl_movielens = DataLoader_Movielens ()
  dl_tmdb = DataLoader_TMDB ()

  if generate_preprocessing:
    dl_tmdb.preprocessing_set()

  movielens_merge = dl_movielens.get_merge_by_item_ids()
  tmdb_preprocessed = dl_tmdb.load_preprocessed()
  tmdb_tags_set =  dl_tmdb.create_tags_for_preprocessed_set()

  dict_dataset = { 
    'movielens-rating-set' : dl_movielens.data_set,
    'movielens-user-set' : dl_movielens.user_set,
    'movielens-item-set' : dl_movielens.item_set,
    'movielens-merge' : movielens_merge,
    'TMDB-preprocessed-set' : tmdb_preprocessed,
    'TMDB-movies-set' : dl_tmdb.movies_set,
    'TMDB-tags-set': tmdb_tags_set
  }
  return dict_dataset

@st.cache_resource()
def prepare_hybrid_rec_system_models (df:pd.DataFrame) -> HybridModel_Weighted:
  data_generator = DataGenerator(dataframe=df)

  # 1. Hybrid for filtrer recommendation-list
  model_svd = Model(
    model=SVD(), # TODO: user-based = True
    name='SVD'
  )
  model_baseline_only = Model(
    model=BaselineOnly(),  # TODO: user-based = False
    name='Baseline Only' 
  )
  model_svd.evaluate(data=data_generator)
  model_baseline_only.evaluate(data=data_generator)
  
  # 2. Init Hybrid
  hybrid = HybridModel_Weighted (
    name='SVD x Baseline Only',
    models=[ model_svd,model_baseline_only ],
    weights=[0.5,0.5]
  )
  _,testset = data_generator.get_train_test_set()
  hybrid.fit(data_generator=data_generator)
  predictions = hybrid.test (testset=testset)

  metrics = Metrics(predictions=predictions)
  metrics = metrics.compute_metrics()

  trainset = hybrid.get_trainset()
  testset = trainset.build_anti_testset()
  predictions:List[Prediction] = hybrid.test(testset)

  return hybrid, metrics, predictions

@st.cache_resource()
def prepare_evaluation_machine_learning_models (df:pd.DataFrame) -> Any:
  results = get_evaluation_and_comparison_of_machine_learning_models(df=df)
  return results

@st.cache_resource()
def load_evaluation_machine_learning_models () -> Any:
  evaluation = pd.read_csv('resources/ml_evaluation.csv')
  return evaluation

@st.cache_resource()
def read_item_names(item_set: pd.DataFrame):
  # item_set tiene itemID y name
  id_to_name = dict(zip(item_set['itemID'],item_set['name']))
  name_to_id = dict(zip(item_set['name'],item_set['itemID']))
  return id_to_name, name_to_id

@st.cache_resource()
def prepare_model_with_full_dataset (rating_set:pd.DataFrame) -> AlgoBase:
  data = Dataset.load_from_df(rating_set[rating_set.columns], Reader(rating_scale=(1,5)) )
  trainset = data.build_full_trainset()
  knn = KNNBaseline( sim_options={ 'name':'pearson_baseline', 'user_based':False }, verbose=False )
  knn.fit(trainset=trainset)
  return knn

# endregion

def get_neighbors (model:AlgoBase, item_id:int, k:int=10) -> List[int]:
  inner_id = model.trainset.to_inner_iid (item_id)
  neighbors = knn.get_neighbors(inner_id,k=k)
  neighbors = (knn.trainset.to_raw_iid (iid) for iid in neighbors)
  return list(neighbors)

def get_est_by_item (predictions:List[Prediction], item_id:int) -> float:
  est_by_id:List = [ ]
  est_by_id = [ est for uid,iid,true_r,est,info in predictions if iid == item_id ]
  return sum(est_by_id)/len(est_by_id)




# Load Datasets
dict_dataset = load_dataset()
rating_set = dict_dataset['movielens-rating-set']
item_set = dict_dataset['movielens-item-set']

mean_rating = dict_dataset['movielens-merge'].groupby('itemID')['rating'].mean().reset_index()

#evaluations = prepare_evaluation_machine_learning_models(df=rating_set)
evaluations = load_evaluation_machine_learning_models()

# Prepare model with 100% Dataset
knn = prepare_model_with_full_dataset(rating_set)

# Prepare hybrid model 
hybrid,metrics,predictions = prepare_hybrid_rec_system_models(rating_set)

# Mapper id:name
id_to_name, name_to_id = read_item_names(item_set)



markdown = """

"""
st.markdown(markdown)



# Config Options
options = name_to_id.keys()

col1_1,col1_2 = st.columns( [1,1] )
with col1_1:
  selection = st.multiselect(label='Select your favorite movies',options=options)
with col1_2:
  selection_k = st.select_slider(label='Number of recommendations', options=[i for i in range(2,11)])
  col3_1,col3_2 = st.columns( [1,1] )
  with col3_1:
    btn_search_neighbors = st.button (label='Compute recommendations')
  with col3_2:
    btn_show_evaluation = st.button (label='Show evaluation')


col2_1,col2_2 = st.columns([1,1])
with col2_1:
  
  if btn_search_neighbors:
    s_neighbors:Dict = { s: get_neighbors(model=knn,item_id=name_to_id[s],k=selection_k) for s in selection }
    merge:Set = set()
    for _,neighbors in s_neighbors.items():
      merge.update(neighbors)

    s_rating:Dict = { m:get_est_by_item(predictions=predictions,item_id=m) for m in merge }

    df_result = pd.DataFrame ({
      "ID": s_rating.keys(),
      "Name": [ id_to_name[key] for key,_ in s_rating.items() ], 
      "Estimation": [ value for _,value in s_rating.items() ],
      "Average Rating": [ mean_rating.loc[ mean_rating['itemID']==key, 'rating'].iloc[0] for key in s_rating.keys() ]
    })
    st.write (df_result.sort_values(by='Estimation',ascending=False).head(selection_k))

with col2_2:
  if btn_show_evaluation:
    st.write (evaluations[ ['Name','Test RMSE', 'Fit Time'] ])



more_comment = """
# TODO
#! Preprocessed-set Display
st.write (df_tmdb_preprocessed_set)
st.write (df_tmdb_tags_set)

movie_recommend = recommend (df=df_tmdb_tags_set, title='Cars', similarity=dict_resources['similarity'])

st.write (movie_recommend)
# Tomar todas las option de df-title-movies y seleccionar para pasarla y que recomiende segun la similitud entre peliculas 
# esto se puede entender como un recomnedador sencillo

"""
