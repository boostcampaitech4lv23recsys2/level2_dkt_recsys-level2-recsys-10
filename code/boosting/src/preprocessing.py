
import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder

def percentile(s):
    return np.sum(s) / len(s)

def preprocessing( df:pd.DataFrame ):
   
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