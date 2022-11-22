import pandas as pd
import os
import random

import torch
import wandb

from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from wandb.xgboost import wandb_callback
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np

import argparse

def feature_engineering(df:pd.DataFrame) -> pd.DataFrame :
    
    #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
    df.sort_values(by=['userID','Timestamp'], inplace=True)
    
    #유저들의 문제 풀이수, 정답 수, 정답률을 시간순으로 누적해서 계산
    df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
    df['user_correct_answer'].fillna(0,inplace=True)
    df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
    df['user_acc'] = df['user_correct_answer']/df['user_total_answer']
    df['user_acc'].fillna(0,inplace=True)

    # testId와 KnowledgeTag의 전체 정답률은 한번에 계산
    # 아래 데이터는 제출용 데이터셋에 대해서도 재사용
    correct_t = df.groupby(['testId'])['answerCode'].agg(['mean', 'sum'])
    correct_t.columns = ["test_mean", 'test_sum']
    correct_k = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum'])
    correct_k.columns = ["tag_mean", 'tag_sum']

    df = pd.merge(df, correct_t, on=['testId'], how="left")
    df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")
    
    return df


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


def main(args):
    wandb.login()

    data_dir = '../../data' # 경로는 상황에 맞춰서 수정해주세요!
    csv_file_path = os.path.join(data_dir, 'train_data.csv') # 데이터는 대회홈페이지에서 받아주세요 :)
    df = pd.read_csv(csv_file_path) 

    df = feature_engineering(df)
    random.seed(args.seed)

    # 유저별 분리
    fold = StratifiedKFold(n_splits = 5, shuffle =True)
    # train, test = custom_train_test_split(df, ratio=args.train_ratio)

    # 사용할 Feature 설정
    FEATS = ['KnowledgeTag', 'user_correct_answer', 'user_total_answer', 
            'user_acc', 'test_mean', 'test_sum', 'tag_mean','tag_sum']

    # X, y 값 분리
    # y_train = train['answerCode']
    # train = train.drop(['answerCode'], axis=1)

    # y_test = test['answerCode']
    # test = test.drop(['answerCode'], axis=1)

    config_defaults = {
        "booster": "gbtree",
        "subsample": 1,
        "seed": 117,
        "test_size": 0.33,
    }

    ##########
    wandb.init(project="xgbc", config=config_defaults)

    # Feel free to change these and experiment !!
    config = wandb.config
    config.colsample_bytree = 0.3
    config.learning_rate = 0.01
    config.max_depth = 15
    config.alpha = 10
    config.n_estimators = 5

    wandb.config.update(config,allow_val_change=True)
    
    config = wandb.config
    ##########
    model = XGBClassifier( booster=config.booster, 
                           max_depth=config.max_depth,
                           learning_rate=config.learning_rate, 
                           subsample=config.subsample )
    X = df.drop(['answerCode'], axis=1)
    y = df['answerCode']

    score_train = []
    score_test = []

    for train_index , test_index in fold.split(X, y):
        X_train,X_test = X.iloc[train_index], X.iloc[test_index]
        y_train,y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train[FEATS], y_train,eval_set=[(X_test[FEATS] , y_test)],eval_metric='auc',verbose=50,early_stopping_rounds= 50)

        y_pred_train = model.predict_proba(X_train[FEATS])[:,1]
        y_pred_test = model.predict_proba(X_test[FEATS])[:,1]
        score_train.append(roc_auc_score( y_train,y_pred_train))
        score_test.append(roc_auc_score( y_test,y_pred_test))
        # make predictions on test

        acc = accuracy_score(y_test, np.where(y_pred_test >= 0.5, 1, 0))
        auc = roc_auc_score( y_test,y_pred_test)
        wandb.log({"valid_accuracy": acc})
        wandb.log({"valid_roc_auc": auc})

    print('\n')
    print('Mean training AUC:',np.mean(score_train))
    print('Mean testing AUC:',np.mean(score_test))
    


    # print(f'VALID AUC : {auc} ACC : {acc}\n')

    # LOAD TESTDATA
    test_csv_file_path = os.path.join(data_dir, 'test_data.csv')
    test_df = pd.read_csv(test_csv_file_path)

    # FEATURE ENGINEERING``
    test_df = feature_engineering(test_df)

    # LEAVE LAST INTERACTION ONLY
    test_df = test_df[test_df['userID'] != test_df['userID'].shift(-1)]

    # DROP ANSWERCODE
    test_ = test_df['answerCode']
    test_df = test_df.drop(['answerCode'], axis=1)

    # MAKE PREDICTION
    total_preds = model.predict(test_df[FEATS])
    # acc_ = accuracy_score(test_, np.where(total_preds >= 0.5, 1, 0))
    # auc_ = roc_auc_score(test_, total_preds)
    # wandb.log({"test_accuracy": acc_})
    # wandb.log({"test_roc_auc": auc_})


    # SAVE OUTPUT
    output_dir = 'output/'
    write_path = os.path.join(output_dir, "submission.csv")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write('{},{}\n'.format(id,p))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--verbose_eval", default=100, type=int, help="verbose_eval")
    parser.add_argument("--num_boost_round", default=500, type=int, help="num_boost_round")
    parser.add_argument("--early_stopping_rounds", default=100, type=int, help="early_stopping_rounds")
    parser.add_argument("--lr", default=1e-2, type=int, help="learnig_rate")
    parser.add_argument("--epochs", default=500, type=int, help="epochs")

    parser.add_argument("--application", default='binary', type=str, help="application") # regression, binary, multiclass
    parser.add_argument("--drop_rate", default=0.1, type=int, help="drop_rate")
    
    parser.add_argument("--train_ratio", default=0.7, type=int, help="train_ratio")

    args = parser.parse_args()

    # os.makedirs(args.model_dir, exist_ok=True)
    main(args)