
import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder

def percentile(s):
    return np.sum(s) / len(s)

def preprocessing( df:pd.DataFrame ):
   
    # #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
    # df.sort_values(by=['userID','Timestamp'], inplace=True)
    
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

def preprocessing_yujin(df):
    
    #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
    df.sort_values(by=['userID','Timestamp'], inplace=True)

    # 문제 푼 시간 : 해당 문제를 푼 시간을 반영하기 위해 shift 를 하면 성능에 영향을 준다는 견해가 있음 
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df["solved_time"] = df.groupby('userID')["Timestamp"].diff().shift(-1)
    df["solved_time"] = df["solved_time"].dt.seconds

    # 이상치 제거 : 2 시간 이상은 평균값으로 설정 
    df.loc[ df["solved_time"] > 14400,"solved_time"] = np.NaN
    fill_mean_func = lambda g: g["solved_time"].fillna(g["solved_time"].median())
    df["solved_time"] = df.groupby('userID').apply(fill_mean_func).reset_index()["solved_time"]

    df["year"]    = df['Timestamp'].dt.year
    df["month"]   = df['Timestamp'].dt.month
    df["day"]     = df['Timestamp'].dt.day
    df["hour"]    = df['Timestamp'].dt.hour
    df["weekday"] = df["Timestamp"].dt.weekday

    # 연도 별 정답률 
    df["correct_ratio_by_year"] = df["year"].map(df.groupby("year").agg({"userID":"count","answerCode":percentile})["answerCode"])
    # 달 별 정답률 
    df["correct_ratio_by_month"] = df["month"].map(df.groupby("month").agg({"userID":"count","answerCode":percentile})["answerCode"])
    # 날짜 별 정답률 
    df["correct_ratio_by_day"] = df["day"].map(df.groupby("day").agg({"userID":"count","answerCode":percentile})["answerCode"])
    # 시간 별 정답률 
    df["correct_ratio_by_hour"] = df["hour"].map(df.groupby("hour").agg({"userID":"count","answerCode":percentile})["answerCode"])
    # 요일 별 정답률 
    df["correct_ratio_by_weekday"] = df["weekday"].map(df.groupby("weekday").agg({"userID":"count","answerCode":percentile})["answerCode"])

    # 시험지 카테고리 : 대분류 
    df['high_test_category'] = df['testId'].str[2]
    df['high_test_category'] = df['high_test_category'].astype('category')

    
    # 시험지 카테고리 : 대분류 
    df['test_category'] = df['testId'].str[7:10]
    df['test_category'] = df['test_category'].astype('category')

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

def preprocessing_hyunho( total:pd.DataFrame ):
    
    ## 유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
    total.sort_values(by=['userID','Timestamp'], inplace=True)
    total['Timestamp'] = pd.to_datetime(total['Timestamp'])
   
    total['same_item_cnt'] = total.groupby(['userID', 'assessmentItemID']).cumcount() + 1
    
    # elapsed
    # total['Timestamp'] = pd.to_datetime(total['Timestamp'])
    # # total['month'] = total['Timestamp'].dt.month
    # total['hour'] = total['Timestamp'].dt.hour
    # diff = total.loc[:, ['userID', 'Timestamp']].groupby('userID').diff().fillna(pd.Timedelta(seconds=0))
    # diff = diff.fillna(pd.Timedelta(seconds=0))
    # diff = diff['Timestamp'].apply(lambda x: x.total_seconds())
    # total['elapsed'] = diff
    
    # 유저, test, same_item_cnt 구분했을 때 문제 푸는데 걸린 시간 > shift, fillna x
    diff_shift = total.loc[:, ['userID', 'testId', 'Timestamp', 'same_item_cnt']].groupby(['userID', 'testId', 'same_item_cnt']).diff().shift(-1)
    diff_shift = diff_shift['Timestamp'].apply(lambda x: x.total_seconds())
    total['solved_time_shift'] = diff_shift
    # total['solved_time_shift'] = total.groupby(['userID', 'testId', 'same_item_cnt'])['solved_time_shift'].apply(lambda x:x.fillna(x.mean()))
    
    # 1. agg 값 구하기
    ## 1-1. 유저/문제/시험지/태그별 평균 정답률
    total['user_avg'] = total.groupby('userID')['answerCode'].transform('mean')
    total['item_avg'] = total.groupby('assessmentItemID')['answerCode'].transform('mean')
    total['test_avg'] = total.groupby('testId')['answerCode'].transform('mean')
    total['tag_avg'] = total.groupby('KnowledgeTag')['answerCode'].transform('mean')

    ## 1-2. 유저/문제/시험지별 평균 풀이시간
    total['user_time_avg'] = total.groupby('userID')['solved_time_shift'].transform('mean')
    total['item_time_avg'] = total.groupby('assessmentItemID')['solved_time_shift'].transform('mean')
    total['test_time_avg'] = total.groupby('testId')['solved_time_shift'].transform('mean')
    total['tag_time_avg'] = total.groupby('KnowledgeTag')['solved_time_shift'].transform('mean')

    # ## 1-3. 유저/문제/시험지/태그별 평균 표준편차
    # total['user_std'] = total.groupby('userID')['answerCode'].transform('std')
    # total['item_std'] = total.groupby('assessmentItemID')['answerCode'].transform('std')
    # total['test_std'] = total.groupby('testId')['answerCode'].transform('std')
    # total['tag_std'] = total.groupby('KnowledgeTag')['answerCode'].transform('std')
    
    # 맞은 사람의 문제별 평균 풀이시간
    total = total.set_index('assessmentItemID')
    total['Item_mean_solved_time'] = total[total['answerCode'] == 1].groupby('assessmentItemID')['solved_time_shift'].mean()
    total = total.reset_index(drop = False)
     
    # 유저/문제/시험지/태그별 표준편차
    total['user_std'] = total.groupby('userID')['answerCode'].transform('std')
    total['item_std'] = total.groupby('assessmentItemID')['answerCode'].transform('std')
    total['test_std'] = total.groupby('testId')['answerCode'].transform('std')
    total['tag_std'] = total.groupby('KnowledgeTag')['answerCode'].transform('std')
    
    ## 1-3. 현재 유저의 해당 문제지 평균 정답률/풀이시간
    total['user_current_avg'] = total.groupby(['userID', 'testId', 'same_item_cnt'])['answerCode'].transform('mean')
    total['user_current_time_avg'] = total.groupby(['userID', 'testId', 'same_item_cnt'])['solved_time_shift'].transform('mean')
    
    # 2. 컬럼 추가
    total['hour'] = total['Timestamp'].dt.hour
    total['hour'] = total['hour'].astype('category')
    
    total['month'] = total['Timestamp'].dt.month
    total['month'] = total['month'].astype('category')
    
    ## 문제 번호 추가
    total['item_num'] = total['assessmentItemID'].str[7:]
    total['item_num'] = total['item_num'].astype('category')
    # in2idx = {v:k for k,v in enumerate(df['item_num'].unique())}
    # df['item_num'] = df['item_num'].map(in2idx)
    
    ## 문제 푼 순서 추가 > 상대적 순서?
    total['item_seq'] = total.groupby(['userID', 'testId', 'same_item_cnt']).cumcount() +1
    total['item_seq'] = total['item_seq'].astype('category')
    # df['item_seq'] = df['item_seq'] - df['item_num'].astype(int)
    # df['item_num'] = df['item_num'].astype('category')
    # is2idx = {v:k for k,v in enumerate(df['item_seq'].unique())}
    # df['item_seq'] = df['item_seq'].map(is2idx)
    
    
    #유저들의 문제 풀이수, 정답 수, 정답률을 시간순으로 누적해서 계산
    # df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
    # df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
    # df['user_acc'] = df['user_correct_answer']/df['user_total_answer']
    
    # diff = df.loc[:, ['userID', 'Timestamp']].groupby('userID').diff().fillna(pd.Timedelta(seconds=0))
    # diff = diff.fillna(pd.Timedelta(seconds=0))
    # diff = diff['Timestamp'].apply(lambda x: x.total_seconds())
    # df['elapsed'] = diff
    
    total['Bigcat'] = total['testId'].str[2]
    total['Bigcat'] = total['Bigcat'].astype('category')
    
    total['smallcat'] = total['testId'].str[7:10]
    total['smallcat'] = total['smallcat'].astype('category')
#     sc2idx = {v:k for k,v in enumerate(df['smallcat'].unique())}
#     df['smallcat'] = df['smallcat'].map(sc2idx)

#     # testId와 KnowledgeTag의 전체 정답률은 한번에 계산
#     # 아래 데이터는 제출용 데이터셋에 대해서도 재사용
#     correct_t = df.groupby(['testId'])['answerCode'].agg(['mean', 'sum'])
#     correct_t.columns = ["test_mean", 'test_sum']
#     correct_k = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum'])
#     correct_k.columns = ["tag_mean", 'tag_sum']

#     df = pd.merge(df, correct_t, on=['testId'], how="left")
#     df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")

    return total