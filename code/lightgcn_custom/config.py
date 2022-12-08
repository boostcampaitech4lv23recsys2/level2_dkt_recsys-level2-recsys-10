# ====================================================
# CFG
# ====================================================
class CFG:
    use_cuda_if_available = True
    user_wandb = True
    wandb_kwargs = dict(project="dkt-gcn")

    # data
    basepath = "/opt/ml/input/data/"
    loader_verbose = True

    # dump
    output_dir = "./output/"
    pred_file = "submission_2.csv"

    # build
    kfold = 1
    embedding_dim = 256  # int
    num_layers = 2  # int
    alpha = None  # Optional[Union[float, Tensor]]
    build_kwargs = {}  # other arguments
    weight = "./weight/best_model"
    # weight = "./weight/best_model.pt"

    # train
    n_epoch = 3000
    early_stop = 3000
    learning_rate = 0.0015
    weight_basepath = "./weight"

    # sweep
    sweep=False
    sweep_count = 20
    sweep_name = 'yujin_test'

sweep_conf = {
    'name' : 'lightgcn_custom',
    'method': 'bayes',
    'metric' : {
        'name': 'auc',
        'goal': 'maximize'   
        },
    'parameters' : {
        'learning_rate': {
            'distribution': 'uniform',
            'min': 0.0013,
            'max': 0.003
        },
        'num_layers': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 4
        },
        'embedding_dim': {
            'distribution': 'int_uniform',
            'min': 2,
            'max': 128
        },
        'dropout':{
            'values': [0.2, 0.3, 0.4, 0.5]
        },

    }
}

logging_conf = {  # only used when 'user_wandb==False'
    "version": 1,
    "formatters": {
        "basic": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "basic",
            "stream": "ext://sys.stdout",
        },
        "file_handler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "basic",
            "filename": "run.log",
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "file_handler"]},
}
