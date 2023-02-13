import os

import torch
import wandb
from args import parse_args
from src import trainer
from src.dataloader import Preprocess
from src.utils import setSeeds

from sklearn.model_selection import KFold
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler


def objective(trial:Trial, train_data, valid_data):
    args.lr = trial.suggest_float('lr',0.001,0.003,step=0.0001)
    # args.max_seq_len = trial.suggest_int('max_seq_len',80,150,step=10)
    # args.stride = trial.suggest_int('stride',60,80,step=10)
    # args.drop_out = trial.suggest_float('drop_out',0.2,0.6,step=0.1)
    args.patience = trial.suggest_int('patience',25,45,step=5)
    args.clip_grad = trial.suggest_int('clip_grad',100,200,step=10)
        
    model = trainer.get_model(args).to(args.device)
    kf_auc = []
        
    auc = trainer.run(args, train_data, valid_data, model, kf_auc)
    
    return auc

def main(args):
    wandb.login()
    
    setSeeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()

    train_data, valid_data = preprocess.split_data(train_data)


    wandb.init(project="dkt", config=vars(args))
    
    if args.split == 'user':
        # model = trainer.get_model(args).to(args.device)
        # kf_auc = []
        # trainer.run(args, train_data, valid_data, model, kf_auc)
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler())
        study.optimize(lambda trial : objective(trial, train_data, valid_data), n_trials=100)
        print('######################################')
        print(f'Best Trial : {study.best_trial.value}')
        print(f'param : {study.best_trial.params}')
        print('######################################')
        # optuna.visualization.plot_param_importances(study)
        # optuna.visualization.plot_optimization_history(study)
        
    elif args.split == 'k-fold':
        # model = trainer.get_model(args).to(args.device)
        n_splits = args.n_splits
        kf_auc = []
        kf = KFold(n_splits=n_splits)

        for idx, (train_idx, test_idx) in enumerate(kf.split(train_data)):
            train_ = torch.utils.data.Subset(train_data, indices = train_idx)
            test_ = torch.utils.data.Subset(train_data, indices = test_idx)

            model = trainer.get_model(args).to(args.device)
            
            trainer.run(args, train_, test_, model, kf_auc, idx+1)
        
        for i in range(n_splits):
            print(f'Best AUC of {i+1} fold : {kf_auc[i]}')
        print(f'Average AUC : {sum(kf_auc)/n_splits:.4f}')

        wandb.log({
            'kfold_avg_valid_auc' : sum(kf_auc)/n_splits
        })
        


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
    
    # main()
    # study = optuna.create_study(direction='minimize', sampler=TPESampler())
    # study.optimize(lambda args : main(), n_trials=10)
    # print(f'Best Trial : {study.best_trial.value}, param : {study.best_trial.params}')
    # optuna.visualization.plot_param_importances(study)
    # optuna.visualization.plot_optimization_history(study)
    
    
    
    #     args = {
#         'seed' : 42,
#         'device' : 'gpu',
#         'data_dir' : '../../data/',
#         'asset_dir' : 'asset/',
#         'file_name' : 'total_data.csv',
#         'model_dir' : 'models/',
#         'model_name' : 'model.pt',
#         'output_dir' : 'output/',
#         'test_file_name' : 'total_data.csv',
#         'model' : 'lastquery',
#         'optimizer' : 'adam',
#         'scheduler' : 'linear_warmup',
        
#         'num_workers' : 1,
#         'max_seq_len' : 100,
#         'hidden_dim' : 256,
#         'n_layers' : 1,
#         'n_heads' : 16,
#         'drop_out' : 0.4,
#         'n_epochs' : 100,
#         'batch_size' : 64,
#         'lr' : Trial().trial.suggest_float('lr',1e-3,3e-3,step=1e-3),
#         'clip_grad' : 100,
#         'patience' : 10,
#         'log_steps' : 50,
#         'window' : True,
#         'shuffle' : False,
#         'stride' : 100,
#         'shuffle_n' : 3,
#         'cate_feats' : ["assessmentItemID", 
#                                 'testId',
#                                 'Bigcat',
#                                 'KnowledgeTag',
#                                 'timeClass',
#                                 'Bigcat_class',
#                        ],
#         'conti_feats' : ['prior_elapsed',
#                                 'user_avg', 'item_avg', 'tag_avg', 'Bigcat_avg',
#                                 'user_std', 'item_std', 'tag_std', 'Bigcat_std',
#                                 'user_retCumacc', 'user_retCount_correct_answer',
#                                 'user_cumacc',
#                                 'elo', 
#                         ],
#         'split' : 'user',
#         'n_splits' : 5,
#     }
    