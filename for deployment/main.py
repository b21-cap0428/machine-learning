# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# This Jupyter notebook will combine the data from survey and data from firestore and use it for training the Machine Learning.

# %%
# set the environment path to find Recommenders
import sys
sys.path.append("../../")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from reco_utils.recommender.rbm.rbm import RBM
from reco_utils.dataset.python_splitters import numpy_stratified_split
from reco_utils.dataset.sparse import AffinityMatrix

from reco_utils.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k

import firebase_admin as fb
from firebase_admin import firestore


def main(HTTP_Arg):
# %% [markdown]
# This Section will take the data from both firestore and survey

# %%
#initialize firebase-admin library
    cred = fb.credentials.Certificate('credential.json')
    fb.initialize_app(cred, {'databaseURL': 'https://keenam-cap0428-default-rtdb.asia-southeast1.firebasedatabase.app'})
    db = firestore.client()
    batch = db.batch()


    # %%
    #Get data from firestore. data will be taken from all documents with that contain field 'foodPreference' and 'drinkPreference' and from a collection
    ratingVal=5
    userID_arr=[]
    movieID_arr=[]
    rating_arr=[]

    actual_user_arr=[]

    users_collection='users'

    docs = db.collection(users_collection).stream()

    for doc in docs:
        if 'foodPreference' in doc.to_dict() and 'drinkPreference' in doc.to_dict() and doc.id != '75oB05myWqAonn0HB4cF':
            username=doc.id
            actual_user_arr.append(username)
            for foodname in doc.to_dict()['foodPreference']:
                if foodname:
                    print(username)
                    print(foodname)
                    print(ratingVal)
                    userID_arr.append(username)
                    movieID_arr.append(foodname)
                    rating_arr.append(ratingVal)
            for foodname in doc.to_dict()['drinkPreference']:
                if foodname:
                    print(username)
                    print(foodname)
                    print(ratingVal)
                    userID_arr.append(username)
                    movieID_arr.append(foodname)
                    rating_arr.append(ratingVal)


    # %%
    #load survey data from github
    Retrieval= 'https://raw.githubusercontent.com/b21-cap0428/machine-learning/main/RetrievalV5.csv'
    data = pd.read_csv(Retrieval)

    # Convert to 32-bit in order to reduce memory consumption 
    data.loc[:, 'rating'] = data['rating'].astype(np.int32) 

    data.head()


    # %%
    #generate new userID from both data. 
    #we will also save this array in order to match the userID from firestore and for training the ML
    last_data_userID=data.tail(1)['userid'].values[0]+1
    numbered_user_arr=[]
    for i in range(len(actual_user_arr)):
        numbered_user_arr.append(i+last_data_userID)


    # %%
    #Create new combined dataframe for ML
    new_data=pd.DataFrame([userID_arr,movieID_arr,rating_arr]).T
    new_data.columns=('userid','foodanddrinkname','rating')
    #matching the generated userID for firestore and for ML
    for i in range(len(actual_user_arr)):
        new_data['userid'].loc[new_data['userid'] == actual_user_arr[i]]=numbered_user_arr[i]
    data=pd.concat([data,new_data], ignore_index=True,axis=0)
    #cast the data into int for training
    data['userid']=data['userid'].astype('int')
    data['rating']=data['rating'].astype('int')


    # %%
    #take data column name to prevent hardcoding (only for the training section)
    userid_colname=data.columns[0]
    item_colname=data.columns[1]
    rating_colname=data.columns[2]


    # %%
    The parts below is being kept intact from 


    # %%
    #to use standard names across the analysis 
    header = {
            "col_user": userid_colname,
            "col_item": item_colname,
            "col_rating": rating_colname,
        }

    #instantiate the sparse matrix generation  
    am = AffinityMatrix(DF = data, **header)

    #obtain the sparse matrix 
    X, _, _ = am.gen_affinity_matrix()


    # %%
    Xtr, Xtst = numpy_stratified_split(X)

    # %% [markdown]
    # Training

    # %%
    #First we initialize the model class
    model = RBM(hidden_units= 1000, training_epoch = 250, minibatch_size= 60, keep_prob=0.9,with_metrics =True)


    # %%
    #Model Fit
    train_time= model.fit(Xtr, Xtst)

    # %% [markdown]
    # predicting

    # %%
    #number of top score elements to be recommended  
    K = 10

    #Model prediction on the test set Xtst. 
    top_k, test_time =  model.recommend_k_items(Xtst)


    # %%
    top_k_df = am.map_back_sparse(top_k, kind = 'prediction')
    test_df = am.map_back_sparse(Xtst, kind = 'ratings')


    # %%
    top_k_df = am.map_back_sparse(top_k, kind = 'prediction')
    test_df = am.map_back_sparse(Xtst, kind = 'ratings')


    # %%
    top_k_df.sort_values('prediction', ascending = False)

    # %% [markdown]
    # Evaluation Metrics

    # %%
    def ranking_metrics(
        data_size,
        data_true,
        data_pred,
        time_train,
        time_test,
        K
    ):

        eval_map = map_at_k(data_true, data_pred, col_user=userid_colname, col_item=item_colname, 
                        col_rating=rating_colname, col_prediction="prediction", 
                        relevancy_method="top_k", k= K)

        eval_ndcg = ndcg_at_k(data_true, data_pred, col_user=userid_colname, col_item=item_colname, 
                        col_rating=rating_colname, col_prediction="prediction", 
                        relevancy_method="top_k", k= K)

        eval_precision = precision_at_k(data_true, data_pred, col_user=userid_colname, col_item=item_colname, 
                        col_rating=rating_colname, col_prediction="prediction", 
                                relevancy_method="top_k", k= K)

        eval_recall = recall_at_k(data_true, data_pred, col_user=userid_colname, col_item=item_colname, 
                        col_rating=rating_colname, col_prediction="prediction", 
                            relevancy_method="top_k", k= K)
                            
        df_result = pd.DataFrame(
            {   "Dataset": data_size,
                "K": K,
                "MAP": eval_map,
                "nDCG@k": eval_ndcg,
                "Precision@k": eval_precision,
                "Recall@k": eval_recall,
                "Train time (s)": time_train,
                "Test time (s)": time_test
            }, 
            index=[0]
        )
        
        return df_result


    # %%
    eval_100k= ranking_metrics(
        data_size = "mv 100k",
        data_true =test_df,
        data_pred =top_k_df,
        time_train=train_time,
        time_test =test_time,
        K =10)

    eval_100k

    # %% [markdown]
    # Send to Firestore

    # %%
    top_k_df.loc[top_k_df[userid_colname] == numbered_user_arr[i]].sort_values('prediction', ascending = False)


    # %%
    #get list of all valid foods and drinks
    Fooddoc = db.collection('DefaultVal').document('FoodDrink')
    PossibleFoodandDrink=Fooddoc.get().to_dict()['Food']+Fooddoc.get().to_dict()['Drink']


    # %%
    for i in range(len(actual_user_arr)):

        recommendation_dict={}
        doc_name=actual_user_arr[i]
        user_db = db.collection(users_collection).document(doc_name)
        for key in user_db.get().to_dict().keys():
            if key in PossibleFoodandDrink:
                recommendation_dict[key]=firestore.DELETE_FIELD

        df_to_convert=top_k_df.loc[top_k_df[userid_colname] == numbered_user_arr[i]].sort_values('prediction', ascending = False)

        
        for j in range(len(df_to_convert)):
            #per_recommendation_dict={}
            #per_recommendation_dict['food_name']=df_to_convert['movieID'].values[j]
            #per_recommendation_dict['rating']=df_to_convert['prediction'].values[j]
            #recommendation_arr.append(per_recommendation_dict)
            recommendation_dict[df_to_convert[item_colname].values[j]]=df_to_convert['prediction'].values[j]
        user_data_dict={'categoryRecommendation': recommendation_dict}
        #print(users_collection)
        #print(doc_name)
        #print(user_data_dict)
        #print(user_data_dict)

        batch.update(user_db,recommendation_dict)
    batch.commit()


