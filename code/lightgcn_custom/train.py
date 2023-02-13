
import pandas as pd
import torch
from config import CFG, logging_conf,sweep_conf
from lightgcn.datasets import prepare_dataset, prepare_dataset_kfold
from lightgcn.models import build, train, train_kfold
from lightgcn.utils import class2dict, get_logger,setSeeds

logger = get_logger(logging_conf)
use_cuda = torch.cuda.is_available() and CFG.use_cuda_if_available
device = torch.device("cuda" if use_cuda else "cpu")
print(device)


def main():
    setSeeds()
    logger.info("Task Started")

    logger.info("[1/1] Data Preparing - Start")

    if 1 == CFG.kfold : 
        train_data, valid_data,test_data, num_info, additional_data = prepare_dataset(
            device, CFG.basepath, verbose=CFG.loader_verbose, logger=logger.getChild("data")
        )

    else:
        train_data, valid_data,test_data, num_info, additional_data = prepare_dataset_kfold(
            device, CFG.basepath, verbose=CFG.loader_verbose, logger=logger.getChild("data")
        )
    
    logger.info("[1/1] Data Preparing - Done")

    logger.info("[2/2] Model Building - Start")

    if CFG.sweep:
        import wandb

        def runner():
            wandb.init(config=class2dict(CFG))
            cur_config = wandb.config

            model = build(
                num_info,
                embedding_dim=cur_config.embedding_dim,
                num_layers=cur_config.num_layers,
                alpha=CFG.alpha,
                logger=logger.getChild("build"),
                **CFG.build_kwargs
            )
            model.to(device)
            wandb.watch(model)

            if CFG.user_wandb:
                wandb.watch(model)

            logger.info("[2/2] Model Building - Done")

            logger.info("[3/3] Model Training - Start")
            # train(
            train_kfold(
                model,
                train_data,
                additional_data,
                valid_data,
                n_epoch=cur_config.n_epoch,
                early_stop = CFG.early_stop,
                learning_rate=cur_config.learning_rate,
                dropout=cur_config.dropout,
                use_wandb= CFG.user_wandb,
                weight=cur_config.weight_basepath,
                logger=logger.getChild("train"),
            )

        sweep_id = wandb.sweep(sweep_conf, entity="recsys-10", project="lightgcn")
        wandb.agent(sweep_id, runner, count=CFG.sweep_count)

    else :
        if CFG.user_wandb:
            import wandb
            wandb.init(**CFG.wandb_kwargs, config=class2dict(CFG))

        model = build(
                num_info,
                embedding_dim=CFG.embedding_dim,
                num_layers=CFG.num_layers,
                alpha=CFG.alpha,
                logger=logger.getChild("build"),
                **CFG.build_kwargs
            )
            
        model.to(device)

        if CFG.user_wandb:
            wandb.watch(model)

        logger.info("[2/2] Model Building - Done")

        logger.info("[3/3] Model Training - Start")
        # train(
        train_kfold(
            model,
            train_data,
            additional_data,
            valid_data,
            n_epoch=CFG.n_epoch,
            early_stop = CFG.early_stop,
            learning_rate=CFG.learning_rate,
            use_wandb=CFG.user_wandb,
            weight=CFG.weight_basepath,
            logger=logger.getChild("train"),
            )

    logger.info("[3/3] Model Training - Done")

    logger.info("Task Complete")


if __name__ == "__main__":
    main()
