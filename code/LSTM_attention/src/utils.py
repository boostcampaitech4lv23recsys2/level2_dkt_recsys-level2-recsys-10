import os
import random

import numpy as np
import torch


def setSeeds(seed=42):

    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
<<<<<<< HEAD:code/LSTM_attention/src/utils.py
=======

    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    
    torch.backends.cudnn.deterministic = True
>>>>>>> 28ab840b4d3f51ec2247dbb31d544f5266139876:code/dkt/src/utils.py
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
