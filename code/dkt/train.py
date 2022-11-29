import os

import torch
import wandb
from args import parse_args
from src import trainer
from src.dataloader import Preprocess
from src.utils import setSeeds

from sklearn.model_selection import KFold


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
        wandb.init(project="dkt", config=vars(args))
        model = trainer.get_model(args).to(args.device)
        kf_auc = []
        trainer.run(args, train_data, valid_data, model, kf_auc)
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
