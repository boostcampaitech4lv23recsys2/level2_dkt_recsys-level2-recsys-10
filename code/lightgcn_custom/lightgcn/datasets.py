import os

import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import random

# train과 test 데이터셋은 사용자 별로 묶어서 분리를 해주어야함
def custom_train_test_split(df, ratio=0.7, split=True):
    
    users = list(zip(df['userID'].value_counts().index, df['userID'].value_counts()))
    random.shuffle(users)
    
    max_train_data_len = ratio*len(df)
    sum_of_train_data = 0
    user_ids =[]

    for user_id, count in users:
        sum_of_train_data += count
        if max_train_data_len < sum_of_train_data:
            break
        user_ids.append(user_id)


    train = df[df['userID'].isin(user_ids)]
    test = df[df['userID'].isin(user_ids) == False]

    #test데이터셋은 각 유저의 마지막 interaction만 추출
    test = test[test['userID'] != test['userID'].shift(-1)]
    return train, test

def prepare_dataset(device, basepath, verbose=True, logger=None):
    data = load_data(basepath)
    preprocessing_data(data)
    train_data, test_data = separate_data(data)
    # add split function 
    train,valid = train_test_split(train_data, test_size=0.2)
    # train,valid = custom_train_test_split(train_data)
    id2index, num_info = indexing_data(data)
    additional_data = get_additional_data_list(data)
    train_data_proc = process_data(train, id2index, device)
    valid_data_proc = process_data(valid, id2index, device)
    test_data_proc = process_data(test_data, id2index, device)

    if verbose:
        print_data_stat(train_data, "Train", logger=logger)
        print_data_stat(test_data, "Test", logger=logger)

    # return train_data_proc, test_data_proc, len(id2index)
    return train_data_proc,valid_data_proc, test_data_proc, num_info,  additional_data


def load_data(basepath):
    path1 = os.path.join(basepath, "train_data.csv")
    path2 = os.path.join(basepath, "test_data.csv")
    data1 = pd.read_csv(path1)
    data2 = pd.read_csv(path2)

    data = pd.concat([data1, data2])
    data.drop_duplicates(
        subset=["userID", "assessmentItemID"], keep="last", inplace=True
    )

    return data


def separate_data(data):
    train_data = data[data.answerCode >= 0]
    test_data = data[data.answerCode < 0]

    return train_data, test_data


def indexing_data( data : pd.DataFrame ):
    
 
    """ 각 userID 와 assessment id 를 고유한 id 로 mapping 한다. 
    
    return
        { id2index : user 와 assessmentItemId mapping 정보 ,
          n_user : 고유한 user id 수  
          n_item : 고유한 assessment item id 수 
        }
        
    """
    
    userid, itemid = (
        sorted(list(set(data.userID))),
        sorted(list(set(data.assessmentItemID))),
    )
    n_user, n_item = len(userid), len(itemid)

    userid_2_index = {v: i for i, v in enumerate(userid)}
    itemid_2_index = {v: i + n_user for i, v in enumerate(itemid)}
    # userid_2_index dict 에 itemid_2_index 를 concat
    id_2_index = dict(userid_2_index, **itemid_2_index)
    
    tagid = sorted(list(set(data.KnowledgeTag)))
    tagid_2_index = {v: i for i, v in enumerate(tagid)}
    data["KnowledgeTag"] = data["KnowledgeTag"].map(tagid_2_index)

    testid = sorted(list(set(data.testId)))
    testid_2_index = {v: i for i, v in enumerate(testid)}
    data["testId"] = data["testId"].map(testid_2_index)

    bigcatid = sorted(list(set(data.big_category)))
    bigcatid_2_index = {v: i for i, v in enumerate(bigcatid)}
    data["big_category"] = data["big_category"].map(bigcatid_2_index)

    
    num_info = {
        "n_user" : n_user,
        "n_item" : n_item,
        "n_tags" : len(tagid),
        "n_testids" : len(testid),
        "n_bigcat" : len(bigcatid),
    }

    return id_2_index, num_info

def preprocessing_data(data):
    
    """ data preprocessing 
    1. KnowledgeTag 가 assessmentItemId 와 1 대 1 매칭되는지 확인
    - 매칭된다면 : 그대로 사용
    - 매칭안된다면 : assessmentItemId 기준으로 knowledgeTag 합치기 , label encoding
    """

    """
    2. 각 assessment 의 대분류 카테고리
    """
    data["big_category"] = data["testId"].apply(lambda x: x[2])


    """ 
    3. 각 assessment 소요 시간 추가 
    4. uesr 별 문제 풀이 소요 시간 추가 
    """
    """
    5. (further more) user - assessmentItemId 에 따른 소요시간 추가 
    """
    data["Timestamp"] = pd.to_datetime( data["Timestamp"] )
    data["solved_time"] = data.groupby('userID')["Timestamp"].diff().shift(-1).dt.seconds
    data.loc[ data["solved_time"] > 14400,"solved_time"] = np.NaN

    fill_mean_func = lambda g: g["solved_time"].fillna(g["solved_time"].median())
    data["solved_time"] = data.groupby('userID').apply(fill_mean_func).reset_index()["solved_time"]

    # normalized_time
    # data["solved_time"] = np.log( data["solved_time"])

    # data.loc[ np.isinf(data['solved_time']),"solved_time"] = 0
    
    pass

def get_additional_data_list(data):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # assessmentItemId 기준으로 group 지어 상위 1 개 데이터만 가져옴 
    question_info = data[['assessmentItemID','testId','big_category', 'KnowledgeTag']].groupby("assessmentItemID").head(1).copy()
    print(data["solved_time"] )
    print(question_info["KnowledgeTag"] )
    return {
        "user" : {},
        "item":{
            "KnowledgeTag" : torch.IntTensor( question_info["KnowledgeTag"] ).to(device),
            "testId" : torch.IntTensor( question_info["testId"] ).to(device),
            "big_category" : torch.IntTensor( question_info["big_category"] ).to(device),
        }
    }


def process_data(data, id_2_index, device):

    ################################ 1. node and label information ################################
    edge, label = [], []
    weight = []
    # user : userID, item : assessmentItemID, value : answerCode
    # 그래프의 연결성을 정의한다. 
    # userID 와 assessmentItemID 가 node 가 되고 각각의 uid, iid 를 이용해 두 node 가 연결되어 있음을 전달.
    for user, item, acode, solved_time in zip(data.userID, data.assessmentItemID, data.answerCode, data.solved_time):
        uid, iid = id_2_index[user], id_2_index[item]
        edge.append([uid, iid])
        label.append(acode)
        weight.append(solved_time)

    edge = torch.LongTensor(edge).T
    label = torch.LongTensor(label)
    weight = torch.FloatTensor(weight)
    return dict(edge=edge.to(device), label=label.to(device), weight = weight.to(device))


def print_data_stat(data, name, logger):
    userid, itemid = list(set(data.userID)), list(set(data.assessmentItemID))
    n_user, n_item = len(userid), len(itemid)

    logger.info(f"{name} Dataset Info")
    logger.info(f" * Num. Users    : {n_user}")
    logger.info(f" * Max. UserID   : {max(userid)}")
    logger.info(f" * Num. Items    : {n_item}")
    logger.info(f" * Num. Records  : {len(data)}")