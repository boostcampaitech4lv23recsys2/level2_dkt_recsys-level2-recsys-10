import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

def prepare_dataset(device, basepath, verbose=True, logger=None):
    data = load_data(basepath)
    train_data, test_data = separate_data(data)

    # add split function 
    train,valid = train_test_split(train_data, test_size=0.2)
    id2index = indexing_data(data)
    additional_infos = get_additional_info_dict(data)
    additional_datas = get_additional_data_list(data)
    train_data_proc = process_data(train, id2index, device)
    valid_data_proc = process_data(valid, id2index, device)
    test_data_proc = process_data(test_data, id2index, device)

    if verbose:
        print_data_stat(train_data, "Train", logger=logger)
        print_data_stat(test_data, "Test", logger=logger)

    # return train_data_proc, test_data_proc, len(id2index)
    return train_data_proc,valid_data_proc, test_data_proc, len(id2index), additional_infos , additional_datas


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


def indexing_data(data):
    userid, itemid = (
        sorted(list(set(data.userID))),
        sorted(list(set(data.assessmentItemID))),
    )
    n_user, n_item = len(userid), len(itemid)

    userid_2_index = {v: i for i, v in enumerate(userid)}
    itemid_2_index = {v: i + n_user for i, v in enumerate(itemid)}
    # userid_2_index dict 에 itemid_2_index 를 concat
    id_2_index = dict(userid_2_index, **itemid_2_index)

    return id_2_index

def preprocessing_data(data):
    pass

def get_additional_data_list(data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return {
        "add_data":{
            "KnowledgeTag" : torch.IntTensor(data["KnowledgeTag"].tolist()).to(device)
        }
    }

def get_additional_info_dict(data):
    return [{"name":"KnowledgeTag", "value":data["KnowledgeTag"].nunique() }]


def process_data(data, id_2_index, device):

    ################################ 1. node and label information ################################
    edge, label = [], []
    # user : userID, item : assessmentItemID, value : answerCode
    for user, item, acode in zip(data.userID, data.assessmentItemID, data.answerCode):
    # for user, item, solved_cnt in zip(data.userID, data.assessmentItemID, data.solved_cnt):
        uid, iid = id_2_index[user], id_2_index[item]
        edge.append([uid, iid])
        label.append(acode)
        # label.append(solved_cnt)

    edge = torch.LongTensor(edge).T
    label = torch.LongTensor(label)
    
    return dict(edge=edge.to(device), label=label.to(device))


def print_data_stat(data, name, logger):
    userid, itemid = list(set(data.userID)), list(set(data.assessmentItemID))
    n_user, n_item = len(userid), len(itemid)

    logger.info(f"{name} Dataset Info")
    logger.info(f" * Num. Users    : {n_user}")
    logger.info(f" * Max. UserID   : {max(userid)}")
    logger.info(f" * Num. Items    : {n_item}")
    logger.info(f" * Num. Records  : {len(data)}")