import pandas as pd
import os
import random

import torch
##import wandb

import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import numpy as np

import argparse

####################################################################
def x_100(value):   #0~100범위로 바꿔주기
    return int(value * 100)

def time_zero(value):   #문제 푼 시간 범위 
    if value < 0:
        return 7
    if value > 3600:
        return 3600
    else:
        return value
def lucky(value):   #찍은 사람 찾기 
    if value < 6:
        return 0
    else:
        return 1

####################################################################

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
    ####################################################################
    #문항별 평균 평점
    ass_aver = df.groupby(['assessmentItemID'])['answerCode'].agg(['mean'])
    #유저별 평균 평점
    user_aver = df.groupby(['userID'])['answerCode'].agg(['mean'])
    df = pd.merge(df, ass_aver, on=['assessmentItemID'], how="left")
    df = pd.merge(df, user_aver, on=['userID'], how="left")
    df = df.rename(columns={'mean_x':'ass_aver'})
    df = df.rename(columns={'mean_y':'user_aver'})
    df['ass_aver'] = df['ass_aver'].apply(x_100)
    df['user_aver'] = df['user_aver'].apply(x_100)

    #난이도
    df['assessment_2'] = df['assessmentItemID'].str[2]
    level = df.groupby(['assessment_2'])['answerCode'].agg(['mean'])
    df = pd.merge(df, level, on=['assessment_2'], how="left")
    df = df.drop(columns = ['assessment_2'])
    df = df.rename(columns={'mean':'level'})
    df['level'] = df['level'].apply(x_100)

    # #찍은 사람
    # df['time'] = df['Timestamp'].str.replace(pat=r'[^A-Za-z0-9]', repl=r'', regex=True)
    # df['time'] = df['time'].astype('int')
    # df['diff'] = df['time'].shift(-1) - df['time']
    # df['diff'] = df['diff'].apply(time_zero)
    # df['diff'] = df['diff'].fillna(7)
    # df['luc'] = df['diff'].apply(lucky)
    # df = df.drop(columns = ['diff','time'])

    # #태그 평점
    # tag_aver = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean'])
    # df = pd.merge(df, tag_aver, on=['KnowledgeTag'], how="left")
    # df = df.rename(columns={'mean':'tag_aver'})
    # df['tag_aver'] = df['tag_aver'].apply(x_100)

    # 문항 번호별 점수 평점 
    df['ass_num'] = df['assessmentItemID'].str[-2:]
    ass_num = df.groupby(['ass_num'])['answerCode'].agg(['mean'])
    df = pd.merge(df, ass_num, on=['ass_num'], how="left")
    df = df.drop(columns = 'ass_num')
    df = df.rename(columns = {'mean' : 'ass_num'})
    df['ass_num'] = df['ass_num'].apply(x_100)

    # 월별 점수 평점 
    df['month'] = df['Timestamp'].str[5:7]
    df['month'] = df['month'].astype('int')
    month_mean = df.groupby(['month'])['answerCode'].agg(['mean'])
    df = pd.merge(df, month_mean, on=['month'], how="left")
    df = df.rename(columns={'mean':'month_mean'})
    df['month_mean'] = df['month_mean'].apply(x_100)

    ####################################################################

    
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
    ##wandb.login()

    data_dir = '/opt/ml/level2_dkt_recsys-level2-recsys-10-2/data' # 경로는 상황에 맞춰서 수정해주세요!
    csv_file_path = os.path.join(data_dir, 'train_data.csv') # 데이터는 대회홈페이지에서 받아주세요 :)
    df = pd.read_csv(csv_file_path) 

    df = feature_engineering(df)
    random.seed(args.seed)

    # 유저별 분리
    train, test = custom_train_test_split(df, ratio=args.train_ratio)

    # 사용할 Feature 설정
    FEATS = ['KnowledgeTag', 'user_correct_answer', 'user_total_answer', 
            'user_acc', 'test_mean', 'test_sum', 'tag_mean','tag_sum',
            'ass_aver','user_aver','level','ass_num','month_mean']
#'luc'
    # X, y 값 분리
    y_train = train['answerCode']
    train = train.drop(['answerCode'], axis=1)

    y_test = test['answerCode']
    test = test.drop(['answerCode'], axis=1)


    lgb_train = lgb.Dataset(train[FEATS], y_train)
    lgb_test = lgb.Dataset(test[FEATS], y_test)

    ##########
    ## wandb.init(project="lgbm", config=vars(args))
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
        ##callbacks=[wandb.lightgbm.wandb_callback()]
    )
    ##wandb.lightgbm.log_summary(model, save_model_checkpoint=True)

    preds = model.predict(test[FEATS])

    acc = accuracy_score(y_test, np.where(preds >= 0.5, 1, 0))
    auc = roc_auc_score(y_test, preds)
    ##wandb.log({"valid_accuracy": acc})
    ##wandb.log({"valid_roc_auc": auc})

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
    output_dir = '/opt/ml/level2_dkt_recsys-level2-recsys-10-2/code/output/'
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