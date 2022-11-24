
import random

# train과 test 데이터셋은 사용자 별로 묶어서 분리를 해주어야함
def custom_train_test_split(df, FEATS, orient_key = 'userID',ratio=0.7, split=True):
    
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