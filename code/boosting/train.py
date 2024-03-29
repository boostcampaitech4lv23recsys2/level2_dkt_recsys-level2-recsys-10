import os
import torch
import wandb
from src.args import parse_args
from src.dataloader import *
from src.preprocessing import *
from src.models import *
from src.utils import setSeeds,save
from src.config import boosting_params


def main(args): 
    setSeeds(args.seed) 

    train_path = os.path.join(args.data_dir, args.file_name) 
    test_path = os.path.join(args.data_dir, args.test_file_name) 

    cur_data_loader = MyDataLoader(train_path, test_path)
    data_collect = cur_data_loader.get_data()

    # FEATS = ['KnowledgeTag', 'user_correct_answer', 'user_total_answer', 
    #         'user_acc', 'test_mean', 'test_sum', 'tag_mean','tag_sum']
    FEATS = ['KnowledgeTag', 'same_item_cnt', 'user_avg', 'item_avg', 'test_avg', 'tag_avg', 'user_time_avg', 'item_time_avg',
       'test_time_avg', 'tag_time_avg', 'user_current_avg', 'user_current_time_avg', 'hour', 'item_num', 'Bigcat','smallcat']
    cur_model = None
    
    if  "XGBClassifier" == args.model :
        cur_model = MyXGBoostClassifier( data_collect, boosting_params[args.model], preprocessing, FEATS)
    
    elif "LGBM" == args.model : 
        
        cur_model = MyLGBM( data_collect, boosting_params[args.model], preprocessing_hyunho, FEATS)
    
    elif "LGBMClassifier" == args.model : 
        
        cur_model = MyLGBMClassifier( data_collect, boosting_params[args.model], preprocessing, FEATS)
    
    elif "CatBoostClassifier" == args.model :
        cur_model = MyCatClassifier( data_collect, boosting_params[args.model], preprocessing_hyunho, FEATS,False)
        

    best_auc = cur_model.train()
    preds    = cur_model.inference()
    save(args,preds,best_auc)

if __name__ == "__main__":
    args = parse_args()
    main(args)
