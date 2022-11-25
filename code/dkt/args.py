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
        # "--file_name", default="train_data_3.csv", type=str, help="train file name"
        "--file_name", default="total_data.csv", type=str, help="train file name"
    )

    parser.add_argument(
        "--model_dir", default="models/", type=str, help="model directory"
    )
    parser.add_argument(
        "--model_name", default="model.pt", type=str, help="model file name"
    )

    parser.add_argument(
        "--output_dir", default="output/", type=str, help="output directory"
    )
    parser.add_argument(
        "--test_file_name", default="total_data.csv", type=str, help="test file name"
    )

    parser.add_argument(
        "--max_seq_len", default=20, type=int, help="max sequence length"       # default : 20 / 1,728
    )
    parser.add_argument("--num_workers", default=1, type=int, help="number of workers")

    # 모델
    parser.add_argument(
        "--hidden_dim", default=64, type=int, help="hidden dimension size"
    )
    parser.add_argument("--n_layers", default=2, type=int, help="number of layers")
    parser.add_argument("--n_heads", default=2, type=int, help="number of heads")
    parser.add_argument("--drop_out", default=0.2, type=float, help="drop out rate")

    # 훈련
    parser.add_argument("--n_epochs", default=20, type=int, help="number of epochs")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
    parser.add_argument("--clip_grad", default=10, type=int, help="clip grad")      # default 10 : gradient exploding 방지
    parser.add_argument("--patience", default=5, type=int, help="for early stopping")

    parser.add_argument(
        "--log_steps", default=50, type=int, help="print log per n steps"
    )

    ### 중요 ###
    parser.add_argument("--model", default="lstm", type=str, help="model type")
    parser.add_argument("--optimizer", default="adam", type=str, help="optimizer type")
    parser.add_argument(
        "--scheduler", default="plateau", type=str, help="scheduler type" # plateau linear_warmup
    )


    # Data Augmentation
    parser.add_argument("--window", default=False, type=bool, help="window") # False 면 augumentation X
    parser.add_argument("--shuffle", default=True, type=bool, help="shuffle")
    parser.add_argument("--stride", default=10, type=int, help="stride")
    parser.add_argument("--shuffle_n", default=10, type=int, help="number of shuffle")

    
    # categorical featurs
    parser.add_argument('--cate_feats', type=list, nargs="+",
                        default=["assessmentItemID", 
                                'testId',
                                'Bigcat',
                                'KnowledgeTag',

                        ],
                        help="category features")

    # continous featurs
    parser.add_argument('--conti_feats', type=list, nargs="+",
                        default=['elapsed',
                                'user_avg', 'item_avg', 'Bigcat_avg', 'tag_avg',
                                'user_std', 'item_std', 'Bigcat_std', 'tag_std',
                                'user_retCumacc', 'item_retCumacc'
                                'user_cumacc', 'user_Bigcat_cumacc', 

                        ], 
                        help = "numeric features")



    args = parser.parse_args()

    return args
