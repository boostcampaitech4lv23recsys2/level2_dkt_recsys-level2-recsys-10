import os
import random
import time

import numpy as np
import torch


def setSeeds(seed=42):

    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

def save(args,preds, auc):

    # SAVE OUTPUT
    file_name = "_".join( [ args.model[:4], str(round(auc,2)) ,str(int(time.time())),"submission.csv" ])
    write_path = os.path.join(args.output_dir,file_name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(preds):
            w.write('{},{}\n'.format(id,p))
