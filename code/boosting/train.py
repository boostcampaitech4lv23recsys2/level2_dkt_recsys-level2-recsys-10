import os

import torch
import wandb
from src.args import parse_args
from src.dataloader import *
from src.preprocessing import *
from src.models import *


def main(args): 

    # setSeeds(args.seed) 
    data_dir = '../../data' # 경로는 상황에 맞춰서 수정해주세요!

    train_path = os.path.join(data_dir, 'train_data.csv') 
    test_path = os.path.join(data_dir, 'test_data.csv') 
    submission_path = os.path.join(data_dir, 'test_data.csv') 

    cur_data_loader = MyDataLoader(train_path, test_path, submission_path, preprocessing)
    preprocessed_data = cur_data_loader.preprocess_data()

    config_defaults ={
            "booster": "gbtree",
            "subsample": 1,
            "seed": 42,
            "test_size": 0.33,
            "max_depth" : 400,
            "learning_rate" : 0.11, 
            "subsample" : 1,
    }
    cur_model = MyXGBoostClassifier( preprocessed_data, config_defaults)
    cur_model.train()
    cur_model.inference()
    cur_model.save()

if __name__ == "__main__":
    args = parse_args()
    main(args)
