import pandas as pd 
import random

class MyDataLoader:

    def __init__( self,
                  train_file:str,
                  test_file:str,
                  submission_file:str,
                  preprocessing_ft)->None :

        self.data_path = {} 
        self.data_path["train_file"] = train_file
        self.data_path["test_file"] = test_file
        self.data_path["submissoin_file"]  = submission_file
        self.preprocessing_ft = preprocessing_ft

        try:
            self.data = {}
            self.data["train"]        =  pd.read_csv(self.data_path["train_file"])
            self.data["test"]         =  pd.read_csv(self.data_path["test_file"])
            self.data["submission"]   =  pd.read_csv(self.data_path["submissoin_file"])
        except Exception as e:
            print("Error ",e)


    def preprocess_data(self) -> dict :
        
        self.data["train"] = self.preprocessing_ft( self.data["train"] )
        self.data["test"]  = self.preprocessing_ft( self.data["test"] )
        
        return self.data


# train과 test 데이터셋은 사용자 별로 묶어서 분리를 해주어야함
def custom_train_test_split(df, orient_key = 'userID',ratio=0.7, split=True):
    
    FEATS = ['KnowledgeTag', 'user_correct_answer', 'user_total_answer', 
            'user_acc', 'test_mean', 'test_sum', 'tag_mean','tag_sum']
    users = list(zip(df[orient_key].value_counts().index, df[orient_key].value_counts()))
    random.shuffle(users)
    
    max_train_data_len = ratio*len(df)
    sum_of_train_data = 0
    user_ids =[]

    for user_id, count in users:
        sum_of_train_data += count
        if max_train_data_len < sum_of_train_data:
            break
        user_ids.append(user_id)


    train = df[df[orient_key].isin(user_ids)]
    valid = df[df[orient_key].isin(user_ids) == False]

    # test데이터셋은 각 유저의 마지막 interaction만 추출
    valid = valid[valid['userID'] != valid['userID'].shift(-1)]

    train_X = train.drop("answerCode",axis = 1 )
    y_train = train["answerCode"]

    valid_X = valid.drop("answerCode",axis = 1 )
    y_valid = valid["answerCode"]

    return train_X[FEATS], y_train, valid_X[FEATS], y_valid
