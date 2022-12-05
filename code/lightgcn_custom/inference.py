import os

import pandas as pd
import numpy as np
import torch
from config import CFG, logging_conf
from lightgcn.datasets import prepare_dataset
from lightgcn.models import build, inference
from lightgcn.utils import get_logger

logger = get_logger(logging_conf)
use_cuda = torch.cuda.is_available() and CFG.use_cuda_if_available
device = torch.device("cuda" if use_cuda else "cpu")

if not os.path.exists(CFG.output_dir):
    os.makedirs(CFG.output_dir)


def main():
    logger.info("Task Started")

    logger.info("[1/4] Data Preparing - Start")
    train_data, valid,test_data,num_info,additional_data= prepare_dataset(
        device, CFG.basepath, verbose=CFG.loader_verbose, logger=logger.getChild("data")
    )
    logger.info("[1/4] Data Preparing - Done")

    logger.info("[2/4] Model Building - Start")

    model_list = [ build(
            num_info,
            embedding_dim=CFG.embedding_dim,
            num_layers=CFG.num_layers,
            alpha=CFG.alpha,
            weight=f"{CFG.weight}_{k_idx}.pt",
            logger=logger.getChild("build"),
            **CFG.build_kwargs
        ) for k_idx in range(CFG.kfold) ]
    # model = build(
    #     num_info,
    #     embedding_dim=CFG.embedding_dim,
    #     num_layers=CFG.num_layers,
    #     alpha=CFG.alpha,
    #     weight=CFG.weight,
    #     logger=logger.getChild("build"),
    #     **CFG.build_kwargs
    # )
    pred_list = [] 
    for model in model_list : 
        model.to(device)
        logger.info("[2/4] Model Building - Done")

        logger.info("[3/4] Inference - Start")
        pred = inference(model, test_data, additional_data,logger=logger.getChild("infer"))
        logger.info("[3/4] Inference - Done")

        logger.info("[4/4] Result Dump - Start")
        pred = pred.detach().cpu().numpy()
        pred_list.append(pred)

    pd.DataFrame({"prediction":  np.mean(pred_list, axis = 0)}).to_csv(
        os.path.join(CFG.output_dir, CFG.pred_file), index_label="id"
    )
    logger.info("[4/4] Result Dump - Done")

    logger.info("Task Complete")


if __name__ == "__main__":
    main()
