import math
import os

import torch
<<<<<<< HEAD:code/LSTM_attention/src/trainer.py
#import wandb
=======
import numpy as np
import wandb
>>>>>>> 28ab840b4d3f51ec2247dbb31d544f5266139876:code/dkt/src/trainer.py
import gc

from .criterion import get_criterion
from .dataloader import get_loaders, data_augmentation
from .metric import get_metric
<<<<<<< HEAD:code/LSTM_attention/src/trainer.py
from .model import LSTM, LSTMATTN, Bert, Saint
=======
from .model import LSTM, LSTMATTN, Bert, LastQuery
>>>>>>> 28ab840b4d3f51ec2247dbb31d544f5266139876:code/dkt/src/trainer.py
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .dataloader import get_loaders, data_augmentation

#def run(args, train_data, valid_data, model):
def run(args, train_data, valid_data, model, kf_auc, kf_n=0):
    # print(args)
    torch.cuda.empty_cache()
    gc.collect()

    augmented_train_data = data_augmentation(train_data, args)
    train_data = augmented_train_data

<<<<<<< HEAD:code/LSTM_attention/src/trainer.py
    train_loader, valid_loader = get_loaders(args, train_data, valid_data)
=======
def run(args, train_data, valid_data, model, kf_auc, kf_n=0):
    torch.cuda.empty_cache()
    gc.collect()

    # data augmentation
    augmented_train_data = data_augmentation(train_data, args)
    if len(augmented_train_data) != len(train_data):
        print(f"Data Augmentation applied. Train data {len(train_data)} -> {len(augmented_train_data)}\n")

    # train_loader, valid_loader = get_loaders(args, train_data, valid_data)
    train_loader, valid_loader = get_loaders(args, augmented_train_data, valid_data)
>>>>>>> 28ab840b4d3f51ec2247dbb31d544f5266139876:code/dkt/src/trainer.py

    # only when using warmup scheduler
    args.total_steps = int(math.ceil(len(train_loader.dataset) / args.batch_size)) * (
        args.n_epochs
    )
    args.warmup_steps = args.total_steps // 10

    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    best_auc = -1
    best_acc = -1
    early_stopping_counter = 0
    for epoch in range(args.n_epochs):

        print(f"Start Training: Epoch {epoch + 1}")

        ### TRAIN
        train_auc, train_acc, train_loss = train(
            train_loader, model, optimizer, scheduler, args
        )

        ### VALID
        auc, acc = validate(valid_loader, model, args)

        ### TODO: model save or early stopping
        # wandb.log(
        #     {
        #         "epoch": epoch,
        #         "train_loss_epoch": train_loss,
        #         "train_auc_epoch": train_auc,
        #         "train_acc_epoch": train_acc,
        #         "valid_auc_epoch": auc,
        #          "valid_acc_epoch": acc,
        #     }
        # )
        if auc > best_auc:
            best_auc = auc
            # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, "module") else model

            name = args.model_name + '.pt'
            if args.split == 'k-fold':
                name = args.model_name + f'_{kf_n}.pt'
            
            # if(kf_n != 0):
            #     name = args.model_name_k_fold + f'_{kf_n}.pt'
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model_to_save.state_dict(),
                },
                args.model_dir,
                name,
            )
            early_stopping_counter = 0
        else:                             # early_stopping
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                print(
                    f"EarlyStopping counter: {early_stopping_counter} out of {args.patience}"
                )
                break

        if acc > best_acc:
            best_acc = acc

        # scheduler
        if args.scheduler == "plateau":
            scheduler.step(best_auc)
        else:
            scheduler.step()
    
    # # save best records
    # report['best_auc'] = best_auc
    # report['best_acc'] = best_acc
    
    kf_auc.append(best_auc)
        

    kf_auc.append(best_auc)


def train(train_loader, model, optimizer, scheduler, args):
    model.train()

    total_preds = []
    total_targets = []
    losses = []
    for step, batch in enumerate(train_loader):
        # input = list(map(lambda t: t.to(args.device), process_batch(batch)))
        input = process_batch(batch, args)
        preds = model(input)
        targets = input[-1]  # correct

        loss = compute_loss(preds, targets)
        update_params(loss, model, optimizer, scheduler, args)

        if step % args.log_steps == 0:
            print(f"Training steps: {step} Loss: {str(loss.item())}")

        # predictions
        preds = preds[:, -1]
        targets = targets[:, -1]

        total_preds.append(preds.detach())
        total_targets.append(targets.detach())
        losses.append(loss)

    total_preds = torch.concat(total_preds).cpu().numpy()
    total_targets = torch.concat(total_targets).cpu().numpy()

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    loss_avg = sum(losses) / len(losses)
    print(f"TRAIN AUC : {auc} ACC : {acc}")
    return auc, acc, loss_avg


def validate(valid_loader, model, args):
    model.eval()

    total_preds = []
    total_targets = []
    for step, batch in enumerate(valid_loader):
<<<<<<< HEAD:code/LSTM_attention/src/trainer.py
        # print('#####################################################')
        # print('batch.shape : ')
        # print(batch.shape)
        input = list(map(lambda t: t.to(args.device), process_batch(batch)))
        #print(input.columns)
        # print('input.shape : ')
        # print(input.shape)
        #input = process_batch(batch)
        
        preds = model(input)
        targets = input[3]  # correct
        #targets = input[-1]
=======
        input = process_batch(batch, args)

        preds = model(input)
        targets = input[-1]  # correct
>>>>>>> 28ab840b4d3f51ec2247dbb31d544f5266139876:code/dkt/src/trainer.py

        # predictions
        preds = preds[:, -1]
        targets = targets[:, -1]

        total_preds.append(preds.detach())
        total_targets.append(targets.detach())

    total_preds = torch.concat(total_preds).cpu().numpy()
    total_targets = torch.concat(total_targets).cpu().numpy()

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)

    print(f"VALID AUC : {auc} ACC : {acc}\n")

    return auc, acc


def inference(args, test_data, model):
    model = load_model(args)
    model.eval()
    _, test_loader = get_loaders(args, None, test_data)

    total_preds = []

    for step, batch in enumerate(test_loader):
<<<<<<< HEAD:code/LSTM_attention/src/trainer.py
        input = list(map(lambda t: t.to(args.device), process_batch(batch)))
        #input = process_batch(batch)
=======
        input = process_batch(batch, args)

>>>>>>> 28ab840b4d3f51ec2247dbb31d544f5266139876:code/dkt/src/trainer.py
        preds = model(input)

        # predictions
        preds = preds[:, -1]
<<<<<<< HEAD:code/LSTM_attention/src/trainer.py
        preds = torch.nn.Sigmoid()(preds)
        preds = preds.cpu().detach().numpy()
        total_preds += list(preds)

    write_path = os.path.join(args.output_dir, "attention_last_2.csv")
=======
        # preds = torch.nn.Sigmoid()(preds)
        preds = preds.cpu().detach().numpy()
        total_preds += list(preds)

    write_path = os.path.join(args.output_dir, f"submission_{args.model_name}.csv")

>>>>>>> 28ab840b4d3f51ec2247dbb31d544f5266139876:code/dkt/src/trainer.py
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write("{},{}\n".format(id, p))


def get_model(args):
    """
    Load model and move tensors to a given devices.
    """
    if args.model == "lstm":
        model = LSTM(args)
    if args.model == "lstmattn":
        model = LSTMATTN(args)
    if args.model == "bert":
        model = Bert(args)
<<<<<<< HEAD:code/LSTM_attention/src/trainer.py
    if args.model == "Saint":
        model = Saint(args)  
=======
    if args.model == "lastquery":
        model = LastQuery(args)

>>>>>>> 28ab840b4d3f51ec2247dbb31d544f5266139876:code/dkt/src/trainer.py

    return model

# 배치 전처리
<<<<<<< HEAD:code/LSTM_attention/src/trainer.py
def process_batch(batch):
    
    (test, question, tag, correct, ass_aver, user_aver, big, 
    past_correct, same_item_cnt, problem_id_mean ,
    month_mean, elo,mask) = batch
    # print('#################################')
    # print(test)
    # print(question)
    # print(tag)
    # print(correct)
    # print(ass_aver)
    # print(user_aver)
    # print(big)
    # print(past_correct)
    # print(same_item_cnt)
    # print(problem_id_mean)
    # print(month_mean)
    # print(elo)
    # print(mask)
    # print('########################################################')
    # print('print batch : ')
    # print(batch)
    #test, question, tag, correct, cls, mask = batch
    
=======
def process_batch(batch, args):

    col = args.columns

    cate_batch =  {col_name : batch[args.cate_loc[col_name]] for col_name in args.cate_loc}
    conti_batch = {col_name : batch[args.conti_loc[col_name]] for col_name in args.conti_loc}

    correct = batch[col['answerCode']]
    mask = batch[-1]

>>>>>>> 28ab840b4d3f51ec2247dbb31d544f5266139876:code/dkt/src/trainer.py
    # change to float
    mask = mask.float()
    correct = correct.float()

    # interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
    interaction = correct + 1  # 패딩을 위해 correct값에 1을 더해준다.
    interaction = interaction.roll(shifts=1, dims=1)
    interaction_mask = mask.roll(shifts=1, dims=1)
    interaction_mask[:, 0] = 0
    interaction = (interaction * interaction_mask).to(torch.int64)

<<<<<<< HEAD:code/LSTM_attention/src/trainer.py
    #  test_id, question_id, tag
    test = ((test + 1) * mask).int()
    question = ((question + 1) * mask).int()
    tag = ((tag + 1) * mask).int()
    ass_aver = ((ass_aver + 1) * mask).float()#.int()
    user_aver = ((user_aver + 1) * mask).float()#.int()
    big = ((big + 1) * mask).int()
    past_correct = ((past_correct + 1) * mask).int()
    same_item_cnt = ((same_item_cnt + 1) * mask).int()
    problem_id_mean = ((problem_id_mean + 1) * mask).float()#.int()
    month_mean = ((month_mean + 1) * mask).float()#.int()
    elo = ((elo + 1) * mask).float()#.int()
    #return (test, question, tag, correct, mask, cls, interaction)
    return (test, question, tag, correct, mask, ass_aver, user_aver,big, 
            past_correct, same_item_cnt, problem_id_mean, month_mean, elo, interaction)
=======

    # category type apply + 1, and mask
    for col_name in cate_batch:
        cate_batch[col_name] = (cate_batch[col_name] + 1 * mask).to(torch.int64).to(args.device)

    # contiuous type apply mask
    for col_name in conti_batch:
        conti_batch[col_name] = (conti_batch[col_name] * mask).to(torch.float32).to(args.device)

    # device memory로 이동
    correct = correct.to(args.device)
    mask = mask.to(args.device)
    interaction = interaction.to(args.device)

    return cate_batch, conti_batch, mask, interaction, correct
    
>>>>>>> 28ab840b4d3f51ec2247dbb31d544f5266139876:code/dkt/src/trainer.py


# loss계산하고 parameter update!
def compute_loss(preds, targets):
    """
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)

    """
    loss = get_criterion(preds, targets)

    # 마지막 시퀀드에 대한 값만 loss 계산
    loss = loss[:, -1]
    # loss = torch.gather(loss, 1, index)
    loss = torch.mean(loss)
    return loss


def update_params(loss, model, optimizer, scheduler, args):
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    optimizer.step()
    optimizer.zero_grad()
    if args.scheduler == "linear_warmup":
        scheduler.step()


def save_checkpoint(state, model_dir, model_filename):
    print("saving model ...\n")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print(model_filename)
    torch.save(state, os.path.join(model_dir, model_filename))


def load_model(args, idx):
    if args.split == 'user':
        model_path = os.path.join(args.model_dir, args.model_name + '.pt')
    elif args.split == 'k-fold':
        model_path = os.path.join(args.model_dir, args.model_name + f'_{idx}.pt')
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # load model state
    model.load_state_dict(load_state["state_dict"], strict=True)

    print("Loading Model from:", model_path, "...Finished.")
    return model

# def load_model(args, idx):
#     if args.split == 'user':
#         model_path = os.path.join(args.model_dir, args.model_name)
#     elif args.split == 'k-fold':
#         model_path = os.path.join(args.model_dir, args.model_name_k_fold + f'_{idx}.pt')
#     print("Loading Model from:", model_path)
#     load_state = torch.load(model_path)
#     model = get_model(args)

#     #load model state
#     model.load_state_dict(load_state["state_dict"], strict=True)

#     print("Loading Model from:", model_path, "...Finished.")
#     return model