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
    
    
    ## 유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
    df.sort_values(by=['userID','Timestamp'], inplace=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    ## 문제 푼 시간 재정의
    # 같은 문제 몇번째 푸는지
    df['same_item_cnt'] = df.groupby(['userID', 'assessmentItemID']).cumcount() + 1
    
    # elapsed
    # total['Timestamp'] = pd.to_datetime(total['Timestamp'])
    # # total['month'] = total['Timestamp'].dt.month
    # total['hour'] = total['Timestamp'].dt.hour
    # diff = total.loc[:, ['userID', 'Timestamp']].groupby('userID').diff().fillna(pd.Timedelta(seconds=0))
    # diff = diff.fillna(pd.Timedelta(seconds=0))
    # diff = diff['Timestamp'].apply(lambda x: x.total_seconds())
    # total['elapsed'] = diff
    
    # 유저, test, same_item_cnt 구분했을 때 문제 푸는데 걸린 시간 > shift, fillna x
    diff_shift = df.loc[:, ['userID', 'testId', 'Timestamp', 'same_item_cnt']].groupby(['userID', 'testId', 'same_item_cnt']).diff().shift(-1)
    diff_shift = diff_shift['Timestamp'].apply(lambda x: x.total_seconds())
    df['solved_time_shift'] = diff_shift
    # total['solved_time_shift'] = total.groupby(['userID', 'testId', 'same_item_cnt'])['solved_time_shift'].apply(lambda x:x.fillna(x.mean()))
    
    # 1. agg 값 구하기
    ## 1-1. 유저/문제/시험지/태그별 평균 정답률
    df['user_avg'] = df.groupby('userID')['answerCode'].transform('mean')
    df['item_avg'] = df.groupby('assessmentItemID')['answerCode'].transform('mean')
    df['test_avg'] = df.groupby('testId')['answerCode'].transform('mean')
    df['tag_avg'] = df.groupby('KnowledgeTag')['answerCode'].transform('mean')
    df['user_avg_bytest'] = df.groupby(['userID','testId'])['answerCode'].transform('mean')

    ## 1-2. 유저/문제/시험지별 평균 풀이시간
    df['user_time_avg'] = df.groupby('userID')['solved_time_shift'].transform('mean')
    df['item_time_avg'] = df.groupby('assessmentItemID')['solved_time_shift'].transform('mean')
    df['test_time_avg'] = df.groupby('testId')['solved_time_shift'].transform('mean')
    df['tag_time_avg'] = df.groupby('KnowledgeTag')['solved_time_shift'].transform('mean')
    
    # 맞은 사람의 문제별 평균 풀이시간
    df = df.set_index('assessmentItemID')
    df['Item_mean_solved_time'] = df[df['answerCode'] == 1].groupby('assessmentItemID')['solved_time_shift'].mean()
    df = df.reset_index(drop = False)
     
    # 유저/문제/시험지/태그별 표준편차
    df['user_std'] = df.groupby('userID')['answerCode'].transform('std')
    df['item_std'] = df.groupby('assessmentItemID')['answerCode'].transform('std')
    df['test_std'] = df.groupby('testId')['answerCode'].transform('std')
    df['tag_std'] = df.groupby('KnowledgeTag')['answerCode'].transform('std')
    
    ## 1-3. 현재 유저의 해당 문제지 평균 정답률/풀이시간
    df['user_current_avg'] = df.groupby(['userID', 'testId', 'same_item_cnt'])['answerCode'].transform('mean')
    df['user_current_time_avg'] = df.groupby(['userID', 'testId', 'same_item_cnt'])['solved_time_shift'].transform('mean')
    
    # 2. 컬럼 추가
    df['hour'] = df['Timestamp'].dt.hour
    df['hour'] = df['hour'].astype('category')
    
    df['month'] = df['Timestamp'].dt.month
    df['month'] = df['month'].astype('category')
    
    ## 문제 번호 추가
    df['item_num'] = df['assessmentItemID'].str[7:]
    df['item_num'] = df['item_num'].astype('category')
    # in2idx = {v:k for k,v in enumerate(df['item_num'].unique())}
    # df['item_num'] = df['item_num'].map(in2idx)
    
    ## 문제 푼 순서 추가 > 상대적 순서?
    df['item_seq'] = df.groupby(['userID', 'testId', 'same_item_cnt']).cumcount() +1
    df['item_seq'] = df['item_seq'].astype('category')
    # df['item_seq'] = df['item_seq'] - df['item_num'].astype(int)
    # df['item_num'] = df['item_num'].astype('category')
    # is2idx = {v:k for k,v in enumerate(df['item_seq'].unique())}
    
    df['Bigcat'] = df['testId'].str[2]
    df['Bigcat'] = df['Bigcat'].astype('category')
    df['Bigcat_avg'] = df.groupby('Bigcat')['answerCode'].transform('mean')
    
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
    # FEATS = ['KnowledgeTag', 'user_correct_answer', 'user_total_answer', 
    #         'user_acc', 'test_mean', 'test_sum', 'tag_mean','tag_sum']
    FEATS = ['KnowledgeTag', 'same_item_cnt', 'user_avg', 'item_avg', 'test_avg', 'tag_avg', 'user_time_avg', 'item_time_avg',
       'test_time_avg', 'tag_time_avg', 'user_current_avg', 'user_current_time_avg', 'hour', 'item_num', 'Bigcat','Bigcat_avg']

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