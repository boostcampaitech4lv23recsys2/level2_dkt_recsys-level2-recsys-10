import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.preprocessing import LabelEncoder


class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self, data, ratio=0.7, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed)  # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train=True):
        cate_cols = ["assessmentItemID", "testId", "KnowledgeTag", "item_num", "item_seq", "big_cat", "small_cat"]

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

        for col in cate_cols:

            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ["unknown"]
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
                le.classes_ = np.load(label_path)

                df[col] = df[col].apply(
                    lambda x: x if str(x) in le.classes_ else "unknown"
                )

            # 모든 컬럼이 범주형이라고 가정
            df[col] = df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test

        def convert_time(s):
            timestamp = time.mktime(
                datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
            )
            return int(timestamp)

        # df["Timestamp"] = df["Timestamp"].apply(convert_time)

        return df

    def __feature_engineering(self, df):
        # TODO
        ## 유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
        df.sort_values(by=['userID','Timestamp'], inplace=True)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        ## 문제 푼 시간 재정의
        # 같은 문제 몇번째 푸는지
        df['same_item_cnt'] = df.groupby(['userID', 'assessmentItemID']).cumcount() + 1
        
        # 유저, test, same_item_cnt 구분했을 때 문제 푸는데 걸린 시간 > shift, fillna x
        diff_shift = df.loc[:, ['userID', 'testId', 'Timestamp', 'same_item_cnt']].groupby(['userID', 'testId', 'same_item_cnt']).diff().shift(-1)
        diff_shift = diff_shift['Timestamp'].apply(lambda x: x.total_seconds())
        df['solved_time_shift'] = diff_shift

        # 1. agg 값 구하기
        ## 1-1. 유저/문제/시험지/태그별 평균 정답률

        ## 1-2. 유저/문제/시험지별 평균 풀이시간

        ## 1-3. 현재 유저의 해당 문제지 평균 정답률/풀이시간
        df['user_current_avg'] = df.groupby(['userID', 'testId', 'same_item_cnt'])['answerCode'].transform('mean')
        df['user_current_time_avg'] = df.groupby(['userID', 'testId', 'same_item_cnt'])['solved_time_shift'].transform('mean')
        
        # 2. 컬럼 추가
        df['item_num'] = df['assessmentItemID'].str[7:]
        df['item_seq'] = df.groupby(['userID', 'testId', 'same_item_cnt']).cumcount() +1
        df["big_cat"] = df["testId"].str[2]
        df["small_cat"] = df["testId"].str[7:10]
        
        return df

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)  # , nrows=100000)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용

        self.args.n_questions = len(
            np.load(os.path.join(self.args.asset_dir, "assessmentItemID_classes.npy"))
        )
        self.args.n_test = len(
            np.load(os.path.join(self.args.asset_dir, "testId_classes.npy"))
        )
        self.args.n_tag = len(
            np.load(os.path.join(self.args.asset_dir, "KnowledgeTag_classes.npy"))
        )
        self.args.n_item_num = len(
            np.load(os.path.join(self.args.asset_dir, "item_num_classes.npy"))
        )
        self.args.n_item_seq = len(
            np.load(os.path.join(self.args.asset_dir, "item_seq_classes.npy"))
        )
        self.args.n_big_cat = len(
            np.load(os.path.join(self.args.asset_dir, "big_cat_classes.npy"))
        )
        self.args.n_small_cat = len(
            np.load(os.path.join(self.args.asset_dir, "small_cat_classes.npy"))
        )

        df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        columns = ["userID", "assessmentItemID", "testId", "answerCode", "KnowledgeTag",
                #    "user_current_avg", "user_current_time_avg",
                   "item_num", "item_seq", "big_cat", "small_cat"
                  ]
        group = (
            df[columns]
            .groupby("userID")
            .apply(
                lambda r: (
                    r["testId"].values,
                    r["assessmentItemID"].values,
                    r["KnowledgeTag"].values,
                    # r["user_current_avg"].values,
                    # r["user_current_time_avg"].values,
                    r["item_num"].values,
                    r["item_seq"].values,
                    r["big_cat"].values,
                    r["small_cat"].values,
                    r["answerCode"].values,
                )
            )
        )

        return group.values

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train=False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])

        test, question, tag, correct = row[0], row[1], row[2], row[3]

        cate_cols = [test, question, tag, correct]

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len :]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        cate_cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            cate_cols[i] = torch.tensor(col)

        return cate_cols

    def __len__(self):
        return len(self.data)


from torch.nn.utils.rnn import pad_sequence


def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])

    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col) :] = col
            col_list[i].append(pre_padded)

    for i, _ in enumerate(col_list):
        col_list[i] = torch.stack(col_list[i])

    return tuple(col_list)


def get_loaders(args, train, valid):

    pin_memory = False
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )

    return train_loader, valid_loader
