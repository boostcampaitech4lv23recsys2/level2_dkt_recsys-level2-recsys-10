import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")

    parser.add_argument("--device", default="gpu", type=str, help="cpu or gpu")

    parser.add_argument(
        "--data_dir",
        default="../../data/",
        type=str,
        help="data directory",
    )
    parser.add_argument(
        "--asset_dir", default="asset/", type=str, help="data directory"
    )

    parser.add_argument(
        "--file_name", default="total_data_hyein6.csv", type=str, help="train file name"
        # "--file_name", default="total_data.csv", type=str, help="train file name"
    )

    parser.add_argument(
        "--model_dir", default="models/", type=str, help="model directory"
    )
    parser.add_argument(
        "--model_name", default="model", type=str, help="model file name"
    )

    # parser.add_argument(
    #     "--model_name_k_fold", default="model", type=str, help="model file name"
    # )

    parser.add_argument(
        "--output_dir", default="output/", type=str, help="output directory"
    )
    parser.add_argument(
        # "--test_file_name", default="total_data.csv", type=str, help="test file name"
        "--test_file_name", default="total_data_hyein6.csv", type=str, help="test file name"
    )



    parser.add_argument(
        "--max_seq_len", default=110, type=int, help="max sequence length"       # default : 20 / 1,728
    )
    parser.add_argument("--num_workers", default=1, type=int, help="number of workers")

    # 모델
    parser.add_argument(
        "--hidden_dim", default=128, type=int, help="hidden dimension size"             # 64
    )
    parser.add_argument("--n_layers", default=1, type=int, help="number of layers")     # 2
    parser.add_argument("--n_heads", default=16, type=int, help="number of heads")      # 2
    parser.add_argument("--drop_out", default=0.5, type=float, help="drop out rate")    # 0.2

    # 훈련
    parser.add_argument("--n_epochs", default=100, type=int, help="number of epochs")   # 20 / 100
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")        # 64
    parser.add_argument("--lr", default=0.003, type=float, help="learning rate")        # 0.0001
    parser.add_argument("--clip_grad", default=100, type=int, help="clip grad")         # 10 : gradient exploding 방지
    parser.add_argument("--patience", default=10, type=int, help="for early stopping")  # 5

    parser.add_argument(
        "--log_steps", default=50, type=int, help="print log per n steps"
    )

    ### 중요 ###
    parser.add_argument("--model", default="bert", type=str, help="model type")         # lstm
    parser.add_argument("--optimizer", default="adam", type=str, help="optimizer type")
    parser.add_argument(
        "--scheduler", default="linear_warmup", type=str, help="scheduler type" # plateau / linear_warmup
    )


    # Data Augmentation
    parser.add_argument("--window", default=True, type=bool, help="window") # False 면 augumentation X
    parser.add_argument("--shuffle", default=False, type=bool, help="shuffle")
    parser.add_argument("--stride", default=101, type=int, help="stride")
    parser.add_argument("--shuffle_n", default=3, type=int, help="number of shuffle")

    
    # categorical featurs
    parser.add_argument('--cate_feats', nargs="+",
                        default=['assessmentItemID',
                                'testId',
                                'KnowledgeTag',
                                'bigcat',
                                'smallcat',
                                # 'timeClass',
                                # 'hour',
                                # 'month',
                                # 'dayofweek',
                                'item_num',
                                # 'item_seq',
                                # ‘Bigcat_class’,
                                'day_diff',
                        ],
                        help="category features")

    # # continous featurs
    # parser.add_argument('--conti_feats', type=list, nargs="+",
    #                     default=['elapsed',
    #                             #'solved_time_prior', #'solved_time',
    #                             # 'user_avg', 'item_avg', 'test_avg', 'bigcat_avg', 'tag_avg',
    #                             # 'user_time_avg', 'item_time_avg', 'test_time_avg', 'bigcat_time_avg', 'tag_time_avg',
    #                             # 'user_std', 'item_std', 'test_std', 'bigcat_std', 'tag_std',
    #                             # 'user_cum_cnt', 'item_cum_cnt', #'test_cum_cnt', 'bigcat_cum_cnt', 'tag_cum_cnt',
    #                             #'user_cor_cum_cnt', 'item_cor_cum_cnt', #'test_cor_cum_cnt', 'bigcat_cor_cum_cnt', 'tag_cor_cum_cnt',
    #                             'user_cum_acc', 'item_cum_acc', #'test_cum_acc', 'bigcat_cum_acc', 'tag_cum_acc',
    #                             # 'test_cum_cnt_per_user', 'bigcat_cum_cnt_per_user', 'tag_cum_cnt_per_user',
    #                             # 'test_cor_cum_per_user', 'bigcat_cor_cum_per_user', 'tag_cor_cum_per_user',
    #                             # 'test_cum_acc_per_user', 'bigcat_cum_acc_per_user', 'tag_cum_acc_per_user',
    #                             'user_cur_avg', 'user_cur_time_avg',
    #                             'user_rec_avg_rolling5', #'user_rec_time_avg_rolling5', 
    #                             # 'user_rec_avg_rolling7', 'user_rec_time_avg_rolling7', 
    #                             # 'user_rec_avg_rolling10', 'user_rec_time_avg_rolling10',
    #                             # 'item_rec_avg_rolling5', 'item_rec_time_avg_rolling5', 
    #                             # 'item_rec_avg_rolling7', 'item_rec_time_avg_rolling7', 
    #                             # 'item_rec_avg_rolling10', 'item_rec_time_avg_rolling10'
    #                     ],
    #                     help = "numeric features")
    # continous featurs
    parser.add_argument('--conti_feats', nargs="+",
                        default=['elapsed',
                                #'solved_time_prior', #'solved_time',
                                'user_avg', 'item_avg', 'test_avg', 'bigcat_avg', 'tag_avg',
                                # 'user_time_avg', 'item_time_avg', 'test_time_avg', 'bigcat_time_avg', 'tag_time_avg',
                                # 'user_std', 'item_std', 'test_std', 'bigcat_std', 'tag_std',
                                # 'user_cum_cnt', 'item_cum_cnt', #'test_cum_cnt', 'bigcat_cum_cnt', 'tag_cum_cnt',
                                #'user_cor_cum_cnt', 'item_cor_cum_cnt', #'test_cor_cum_cnt', 'bigcat_cor_cum_cnt', 'tag_cor_cum_cnt',
                                #'user_cum_acc', 'item_cum_acc', 'test_cum_acc', #'bigcat_cum_acc', 'tag_cum_acc',
                                # 'test_cum_cnt_per_user', 'bigcat_cum_cnt_per_user', 'tag_cum_cnt_per_user',
                                # 'test_cor_cum_per_user', 'bigcat_cor_cum_per_user', 'tag_cor_cum_per_user',
                                'test_cum_acc_per_user', #'bigcat_cum_acc_per_user', 
                                'tag_cum_acc_per_user',
                                # 'user_cur_avg', #'user_cur_time_avg',
                                'user_rec_avg_rolling5', #'user_rec_time_avg_rolling5', 
                                # 'user_rec_avg_rolling7', 'user_rec_time_avg_rolling7', 
                                # 'user_rec_avg_rolling10', 'user_rec_time_avg_rolling10',
                                'item_rec_avg_rolling5', #'item_rec_time_avg_rolling5', 
                                # 'item_rec_avg_rolling7', 'item_rec_time_avg_rolling7', 
                                # 'item_rec_avg_rolling10', 'item_rec_time_avg_rolling10'
                                'elo_assessmentItemID', #'elo_testId', 'elo_KnowledgeTag'
                        ],
                        help = "numeric features")

    # k-fold
    parser.add_argument("--split", default="user", type=str, help="data split strategy")
    parser.add_argument("--n_splits", default=5, type=str, help="number of k-fold splits")

    # hyperopt
    parser.add_argument("--hyperopt", default=False, type=bool, help="hyperopt")

    args = parser.parse_args()

    return args
