import pandas as pd
import os
import random

import torch
import wandb

import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import numpy as np

import argparse

def feature_engineering(df):
    
    #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
    df.sort_values(by=['userID','Timestamp'], inplace=True)
    
    #유저들의 문제 풀이수, 정답 수, 정답률을 시간순으로 누적해서 계산
    df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
    df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
    df['user_acc'] = df['user_correct_answer']/df['user_total_answer']

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
    train, test = custom_train_test_split(df, ratio=args.train_ratio)

    # 사용할 Feature 설정
    FEATS = ['KnowledgeTag', 'user_correct_answer', 'user_total_answer', 
            'user_acc', 'test_mean', 'test_sum', 'tag_mean','tag_sum']

    # X, y 값 분리
    y_train = train['answerCode']
    train = train.drop(['answerCode'], axis=1)

    y_test = test['answerCode']
    test = test.drop(['answerCode'], axis=1)


    lgb_train = lgb.Dataset(train[FEATS], y_train)
    lgb_test = lgb.Dataset(test[FEATS], y_test)

    ##########
    wandb.init(project="lgbm", config=vars(args))
    ##########
    model = lgb.train(
        {'objective': args.application,
        'learning_rate':args.lr,
        'drop_rate':args.drop_rate,
        'num_iterations':args.epochs
        }, 
        lgb_train,
        valid_sets=[lgb_train, lgb_test],
        verbose_eval=args.verbose_eval,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        valid_names=('validation'),
        callbacks=[wandb.lightgbm.wandb_callback()]
    )
    wandb.lightgbm.log_summary(model, save_model_checkpoint=True)

    preds = model.predict(test[FEATS])

    acc = accuracy_score(y_test, np.where(preds >= 0.5, 1, 0))
    auc = roc_auc_score(y_test, preds)
    wandb.log({"valid_accuracy": acc})
    wandb.log({"valid_roc_auc": auc})

    print(f'VALID AUC : {auc} ACC : {acc}\n')




    # LOAD TESTDATA
    test_csv_file_path = os.path.join(data_dir, 'test_data.csv')
    test_df = pd.read_csv(test_csv_file_path)

    # FEATURE ENGINEERING
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